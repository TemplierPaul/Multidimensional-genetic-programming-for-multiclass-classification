class AFP_Program(Program):
    def __init__(self, variables, instructions=None, model=None, age=0):
        super().__init__(variables=variables, instructions=instructions, model=model)
        self.age = age
        self.AFP_fitness=0
        
    def __str__(self):
        """
        Description of the object: returnse the list of instructions and the detailed Var objects. 
        
        :return: String
        """
        s = "AFP_Program" + super().__str__()
        s +='\n' + self.age
        
    def from_Program(self, p, age=0):
        """
        Converts a Program object into an AFP_Program object with similar characteristics
        
        :return: AFP_Program
        """
        new = AFP_Program(variables=p.variables, instructions=p.instructions, model=p.model, age=age)
        new.get_features()
        new.fitness = p.fitness
        new.df = p.df
        return new
    
    def copy(self):
        """
        Copies self into a new independant AFP_Program object.

        :return: AFP_Program
        """
        new = super().copy()
        new = self.from_Program(new, age=self.age)
        return new
    
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
        c1, c2 = super().crossover(other, verbose)
        c1 = self.from_Program(c1, age=max(self.age, other.age))
        c2 = self.from_Program(c2, age=max(self.age, other.age))
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
            r = AFP_Program(variables=self.variables, model=m).generate(len(self.instructions))
            self.instructions = self.crossover(r)[0].instructions
            if verbose: print('Mutated feature:\n', split_features(a.instructions), '\n')

        self.get_features()
        return self
    
    def AFP_dominates(self, other):
        """
        Pareto-dominance wrt age and fitness: returns True if self dominates other, False if not.
        """
        or_eq = (self.fitness >= other.fitness) and self.age <= other.age
        strict = (self.fitness > other.fitness) or self.age < other.age
        return or_eq and strict
        