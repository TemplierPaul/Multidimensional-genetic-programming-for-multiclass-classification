class Program(object):
    def __init__(self, variables, instructions=None, model=None):
        self.instructions = instructions 
        self.variables = variables
        self.vars = None
        self.model=model
        self.fitness = None
        self.df = None

    def __str__(self):
        """
        Description of the object: returnse the list of instructions and the detailed Var objects. 
        
        :return: String
        """
        s = ''
        if self.instructions is None:
            s += 'Empty program'
        else:
            s += str(self.instructions)
        s += '\n'
        if self.vars is None:
            s += 'No variables computed'
        else:
            for i in self.vars:
                s += repr(i) + ' \n'
        return s

    def copy(self):
        """
        Copies self into a new independant Program object.

        :return: Program
        """
        new = Program(variables=self.variables, instructions=self.instructions, model=clone(self.model))
        new.get_features()
        new.fitness = self.fitness
        return new      

    def generate(self, size, to_self=True):
        """
        Generates random instructions list
        
        :param: size Int
            number of elements in the list
        :param: to_self Boolean
            If True, stores the list in self.instructions.
            Else, returns it
        :return: self or list of String
        """
        dispo = 0
        l = []
        while len(l) < size:
            if rd.random() > 0.5:
                l.append(np.random.choice(list(self.variables), 1)[0])
                dispo += 1
            else:
                if rd.random() > 0.5 and dispo >= 1:
                    l.append(np.random.choice(list(Var.FUNCTIONS[1]), 1)[0])
                elif dispo >= 2:
                    l.append(np.random.choice(list(Var.FUNCTIONS[2]), 1)[0])
                    dispo -= 1
        if to_self:
            self.instructions = l
            return self
        else:
            return l

    def get_features(self, verbose=False):
        """
        Generates a list of Var objects corresponding to the instructions list.
        
        :param: verbose Boolean
            If True, prints details for debug
        :return: self
        """
        assert self.instructions is not None
        feat = []
        self.vars = []
        for i in self.instructions:
            if verbose: print(i, [str(f) for f in feat])
            if i in self.variables:
                feat.append(self.variables[i])
            elif i in Var.FUNCTIONS[1]:
                assert len(feat) >= 1
                a = feat.pop(-1)
                feat.append(Var.FUNCTIONS[1][i](a))
            elif i in Var.FUNCTIONS[2]:
                assert len(feat) >= 2
                b = feat.pop(-1)
                a = feat.pop(-1)
                feat.append(Var.FUNCTIONS[2][i](a, b))
        if verbose: print([str(f) for f in feat], '\n')
        self.vars = feat
        return self

    def crossover(self, other, verbose=False):
        """
        Implements the Crossover operator between self and another Program object.
        
        :param: other Program
            The other parent
        :param: verbose Boolean
            If True, prints details for debug
        :return: tuple of 2 Program
            Children generated from the crossover
        """
        assert self.instructions is not None
        assert other.instructions is not None

        # Choice between 2 equally probable actions:
        
        if rd.random() > 0.5:  # Perform sub-tree crossover on root nodes
            # 1 full feature is selected in self and 1 in other, and they are swapped

            a, b = split_features(self.instructions), split_features(other.instructions)  # Features lists
            ra, rb = rd.randint(0, len(a) - 1), rd.randint(0, len(b) - 1)  # Indices of features to swap
            a[ra], b[rb] = b[rb], a[ra] #Swapping the features
            
            # Generating new instructions lists
            a = np.concatenate(a).tolist()
            b = np.concatenate(b).tolist()

            if verbose: print('Crossover on root nodes', ra, rb)

        else:  # Perform sub-tree crossover on random nodes

            # Genetic material is copied to be used without side effects
            a, b = self.instructions.copy(), other.instructions.copy()

            # In each instructions list, a node is selected at random
            # The subtrees defined by these nodes are then swapped
            ra = rd.randint(1, len(a))
            size_a = len(list(split_features(a[:ra])[-1]))

            rb = rd.randint(1, len(b))
            size_b = len(list(split_features(b[:rb])[-1]))

            if verbose:
                print('Crossover on random nodes', ra - size_a, ':', ra, '|', rb - size_b, ':', rb)
                print(a[ra - size_a:ra], b[rb - size_b:rb])

            a[ra - size_a:ra], b[rb - size_b:rb] = b[rb - size_b:rb], a[ra - size_a:ra]

        # 2 children are then produces with the new instructions
        if self.model is None:
            c1 = Program(variables=self.variables,
                     instructions=a).get_features()
            c2 = Program(variables=self.variables,
                         instructions=b).get_features()
        else:
            c1 = Program(variables=self.variables,
                         instructions=a, 
                        model = clone(self.model)).get_features()
            c2 = Program(variables=self.variables,
                         instructions=b,
                        model = clone(self.model)).get_features()
        return c1, c2

    def mutate(self, add_len=5, verbose=False):
        """
        Implements the Mutation operator on self.
        
        :param: add_len Int
            Number of instructions added in the case of mutation by addition.
        :param: verbose Boolean
            If True, prints details for debug
        :return: self
        """
        assert self.instructions is not None

        # If there is only 1 feature, add one or mutate it with equal probabilities
        # If there are multiple features, add one, delete one, or mutate one with equal probabilities
        r = rd.random()
        if len(split_features(self.instructions)) <= 1:
            r = r * 2 / 3 + 1 / 3

        if r < 1 / 3:  # Delete a feature (selected at random)
            # Features are split, one is deleted and the instructions are reformed from the rest
            l = split_features(self.instructions)
            r = rd.randint(0, len(l) - 1)
            l.pop(r)
            self.instructions = np.concatenate(l).tolist()
            if verbose: print('Deleted feature', r, '\n')

        elif r < 2 / 3:  # Add a random feature
            l = self.generate(add_len, to_self=False)  # Random instructions : possibly multiple features
            f = split_features(l)[0]  # New subtree : 1 feature
            self.instructions += list(f)
            if verbose: print('Added feature:', l, '\n', split_features(l), '\n', f, '\n')

        else:  # Change a subtree for a random subtree
            # It is equivalent to replacing self by the crossover of self and a random tree
            if self.model is None:
                m=None
            else:
                m=clone(self.model)
            r = Program(variables=self.variables, model=m).generate(len(self.instructions))
            self.instructions = self.crossover(r)[0].instructions
            if verbose: print('Mutated feature:\n', split_features(a.instructions), '\n')

        self.get_features()
        return self

    def compute(self, data):
        """
        Projects a dataset into a new features space defined by the list of instructions and stores it in self.df
        
        :param: data pandas.DataFrame
            DataFrame to transform
        :return: self
        """
        self.get_features()
        self.df = pd.DataFrame()
        f=[] # Filter for duplicates. For each value, False means the corresponding feature is a duplicate of another, hence needs to be removed. 
        
        # For each variable, compute its value for the provided data and concatenate all in a DataFrame
        for v in self.vars:
            if str(v) in self.df.columns:
                f.append(False)
            else:
                self.df[str(v)] = v.compute(data)
                f.append(True)
        self.trim(remove=True, duplicate=f)
        self.get_features()
        return self

    def trim(self, remove=False, duplicate=None):
        """
        Makes the computed DataFrame acceptable for ML training by removing inf or NaN values and constant columns.
        
        
        :param: remove Boolean
            If True, removes the features from the list of instructions
            Else, only modifies self.df
        :param: duplicate list of Boolean
            Filter with a size equal to the number of features. For each value, False means the corresponding feature is a duplicate of another, hence needs to be removed. 
        :return: self
        """
        def filter_out(f):
            # Filters the list of instructions according to the filter f passed as argument
            if sum(f) != len(f):
                s = np.array(split_features(self.instructions))             
                s = s[np.array(f)].tolist()
                self.instructions = np.concatenate(s).tolist()
                self.get_features()
                return True
            return False
        
        if duplicate is not None:
            filter_out(duplicate)
        
        assert self.df is not None
        if len(self.df.columns) > 0:
            # Check for inf values
            a = (np.isfinite(self.df).sum() == len(self.df))
            # Check for NaN values
            b = (np.sum(self.df.isnull())==0)
            # Check for constant columns
            c = (np.std(self.df) > 0)
            # Combined filter
            f = (a & b & c).tolist()
            # List of features to keep
            l = list(self.df.columns[f])
            self.df = self.df.loc[:, l]
            
            if remove:
                if sum(f)!= len(f):
                    filter_out(f)
      
        else:
            raise Exception('No columns')
        
        return self.df.columns

    def plot(self, X, y):
        """
        Plots all the distributions of pairs of features from a transformed dataset.
        
        :param: X pandas.DataFrame
            Features of the data to transform    
        :param: y numpy.array or list
            Classification targets for the X dataset
        :return: self
        """
        self.compute(data=X)
        self.df['target'] = y
        sns.pairplot(self.df, vars=self.df.columns[:-1], hue='target')
        return self
    
    def dominates(self, other):
        """
        Pareto-dominance wrt fitness and intelligibility: returns True if self dominates other, False if not.
        """
        or_eq = (self.fitness >= other.fitness) and len(self.instructions) <= len(other.instructions)
        strict = (self.fitness > other.fitness) or len(self.instructions) < len(other.instructions)
        return or_eq and strict