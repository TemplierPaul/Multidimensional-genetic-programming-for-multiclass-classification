class Population(object):
    def __init__(self, variables=None, data=None, init_size=10, pop_size=100):
        self.variables = variables # Dict {String : Var} linking var names to the objects (one for each initial feature)
        self.model = NearestCentroid(metric='euclidean') # Classifier
        if data is not None:
            self.import_data(data)
        else:
            self.dataset = {'data': None,
                            'target': None,
                            'feature_names': None}

        # List of programs used as a population of the Genetic Programming algorithm
        self.programs = [Program(variables=self.variables, model=clone(self.model)).generate(init_size).get_features() for _ in range(pop_size)]
        
        self.archive = []
        self.params = {
            'selection_method': 'tournament',
            'crossover': 0.45,
            'mutation': 0.45,
            'tournament': {
                'test_size': 0.33
            }
        }

    def __repr__(self):
        """
        Description of the population 
        
        :return: String
        """
        s = 'Population\n'
        s += 'Size: ' + str(len(self.programs)) + '\n'
        s += 'Model: ' + str(self.model) + '\n'
        if self.dataset['data'] is not None:
            s += 'Data: ' + str(self.dataset['data'].shape) + '\n'
            s += 'Target:' + str(self.dataset['target'].shape) + '\n'
            s += 'Features:' + str(self.dataset['feature_names']) + '\n'
        else:
            s += 'No Dataset\n'
        f = [p.fitness for p in self.programs]
        if None in f:
            self.evaluation(verbose=False, plot=False)
            f = [p.fitness for p in self.programs]
        s += 'Fitness: Mean ' + str(np.mean(f)) + ' | Std ' +  str(np.std(f)) + ' | Max ' +  str(np.max(f)) + '\n'
        return s

    def __str__(self):
        """
        Description of the population
        
        :return: String
        """
        return repr(self)

    def import_data(self, data):
        """
        Imports data from a pandas.DataFrame, from an object loded with sklearn.dataset, or from a dict with 'data', 'target' and 'feature_names' keys.  
        
        :return: self
        """
        self.dataset = {'data': None, # The data
                        'target': None, # Labels for classification
                        'feature_names': None} # Column names

        # Input type: pandas DataFrame
        if type(data) == pd.core.frame.DataFrame:
            print("DF")
            assert 'target' in data.columns
            self.dataset['target'] = data['target']
            data = data.drop(columns=['target'])
            self.dataset['feature_names'] = data.columns
            self.dataset['data'] = data
        
        # Input type: sklearn.utils.Bunch, as returned by sklearn.datasets.load_iris() for example
        elif type(data) == sklearn.utils.Bunch:
            dataset = {
                'data': data['data'],
                'target': data['target'],
                'feature_names': data['feature_names']
            }
            data = dataset

        # Input type: dict {'data':np.array OR list, 'target':np.array OR list, 'feature_names':np.array OR list}
        if type(data) == dict:
            assert 'data' in data
            self.dataset['data'] = data['data']

            assert 'target' in data
            self.dataset['target'] = data['target']

            if 'feature_names' not in data:
                f = [str(i) for i in range(len(data['data'][0]))]
                self.dataset['feature_names'] = f
            else:
                self.dataset['feature_names'] = data['feature_names']

        # Creation of the variables dictionary
        self.variables = {}
        for n in self.dataset['feature_names']:
            self.variables[n] = Var(name=n)

        return self

    def evaluation(self, verbose=False, plot=True):
        """
        Computation of the fitness of each program in the population 
        
        :return: self
        """
        if verbose: print('Evaluation')
            
        # Evaluation on 1/3 of the dataset after training on 2/3
        df = pd.DataFrame(self.dataset['data'], columns=self.dataset['feature_names'])
        
        for i in range(len(self.programs)):
            p = self.programs[i]
            df_p = p.compute(df).df # Transformation of the dataset int oa new feature space
            # Definition of training and testing sets:
            X_train_p, X_test_p, y_train, y_test = train_test_split(df_p,
                                                            self.dataset['target'],
                                                            test_size=self.params['tournament']['test_size'],
                                                            random_state=42)

            if verbose: print(str(X_train_p.shape) +' ' + str(y_train.shape))
            
            # If there is no data, set the fitness to -1
            if np.size(X_train_p, 1) == 0:
                p.fitness = -1
            else:
                try:
                    p.model.fit(X_train_p, y_train)
                    p.fitness = p.model.score(X_test_p, y_test) # The fitness is the performance of the model trained with the new features
                except:
                    print('\n', p)
                    print(X_train_p.shape, y_train.shape)
                    print(X_train_p.columns, X_test_p.columns)
                    raise Exception
            
                if verbose: print(p.fitness)
                    
        if plot: self.plot_fitness()
        self.update_archive() # Update the archive for the best solutions
        return self
    
    def tournament_selection(self, verbose=False, plot=True):
        """
        Implements Tournament Selection, Mutation, Crossover, and Reproduction steps of the Genetic Algorithm.  
        
        :return: self
        """
        # Evaluation
        self.evaluation(verbose=verbose, plot=plot)
        
        # New population
        new_pop = []
        i_max = len(self.programs) - 1  # Max index in self.programs

        # Number of new programs from crossover
        n_cross = math.floor(len(self.programs) * self.params['crossover'])
        # Number of new programs from mutation
        n_mut = math.floor(len(self.programs) * self.params['mutation'])
        # Number of new programs from reproduction
        n_reprod = len(self.programs) - n_cross - n_mut
        
        if verbose: print('Crossover:', n_cross, ' | Mutation;', n_mut, ' | Reproduction:', n_reprod)
        
        if verbose: print('Crossover')
        for k in range(n_cross):
            # Tournament for parent 1
            pa, pb = self.programs[rd.randint(0, i_max)], self.programs[rd.randint(0, i_max)]
            if pa.fitness >= pb.fitness:
                p1 = pa
            else:
                p1 = pb

            # Tournament for parent 2
            pa, pb = self.programs[rd.randint(0, i_max)], self.programs[rd.randint(0, i_max)]
            if pa.fitness >= pb.fitness:
                p2 = pa
            else:
                p2 = pb

            # Add only one children of p1 and p2 to the new population
            # Since c1 and c2 are complementary wrt. p1 and p2, we only keep one of them to diversify the population. 
            c1, c2 = p1.crossover(p2)
            new_pop.append(c1)

        if verbose: print('Mutation')
        for k in range(n_mut):
            # Tournament for initial program to be mutated
            pa, pb = self.programs[rd.randint(0, i_max)], self.programs[rd.randint(0, i_max)]
            if pa.fitness >= pb.fitness:
                p = pa.copy()
            else:
                p = pb.copy()

            # Mutate a copy of the selected program and add it to the new population
            p.mutate(add_len=int(len(self.programs)) / 10)
            new_pop.append(p)

        
        if n_reprod > 0:
            if verbose: print('Reproduction:', n_reprod)
            # Copy the best fit elements into the new programs list
            sorted_progs = sorted(self.programs, key=lambda x: x.fitness, reverse=True)
            for k in range(n_reprod):
                new_pop.append(sorted_progs[k])

        self.programs = new_pop
        return self

    def plot_fitness(self):
        """
        Plots the fitness distribution in the population. Prints its mean, standard deviation and maximum value.
        
        :return: self
        """
        fig = plt.figure(figsize=(8, 8))
        f = [p.fitness for p in self.programs]
        plt.hist(f, bins=int(len(self.programs)/5))
        plt.xlim(0, 1)
        plt.show()
        print('Mean', np.mean(f), ' | Std', np.std(f), ' | Max', np.max(f))
        return self
    
    def get_pareto(self, l):
        """
        Gets the Pareto front from a list l
        
        :return: list of the solutions on the Pareto Front
        """
        pareto = [] # Solutions on the Pareto front
        for p in l:
            dominated = sum([a.dominates(p) for a in l]) # Number of other solutions that dominate p
            if dominated == 0: # If p is not domnated, p is on the Pareto front
                pareto.append(p.copy())
        return pareto
    
    def update_archive(self):
        """
        Add the dominating solutions from the population to the archive, and update the archive by removing the dominated solutions.
        
        :return: self
        """
        l = self.get_pareto(self.programs)
        self.archive = self.get_pareto(self.archive + l)
        return self
    
    def get_archive(self):
        """
        Plots the archive on a (fitness, length) scatter plot.
        The "length" axis represents the length of the instructions list.
        
        :return: self
        """
        f = [i.fitness for i in self.archive]
        l = [len(i.instructions) for i in self.archive]
        plt.scatter(f, l)
        plt.xlabel('Fitness')
        plt.ylabel('Length')
        plt.title('Final archive')
        plt.show()
        return self

    def final_selection(self, val_data):
        """
        Implements the final selection on a validation set and returns the best performing solution.  
        
        :return: Program
            The best performing Program on the validation set from the archive
        """
        # Final Evaluation
        df = pd.DataFrame(self.dataset['data'], columns=self.dataset['feature_names'])
        y_train = self.dataset['target']
        
        val_df = pd.DataFrame(val_data.drop(columns=['target']), columns=self.dataset['feature_names'])
        y_val = val_data['target']
        total_df = pd.concat([df, val_df])
        
        for p in self.archive:
            total_df_p = p.compute(total_df).df
            train_df_p = total_df_p.iloc[:len(df)]
            val_df_p = total_df_p.iloc[len(df):]
            
            if np.size(train_df_p, 1) == 0:
                p.fitness = -1
            else:
                p.model.fit(train_df_p, y_train)
                p.fitness = p.model.score(val_df_p, y_val)
        
        # Selection
        print('>> Evaluating on the validation set')
        self.get_archive()
        self.archive.sort(key=lambda x:x.fitness, reverse=True)
        print("\nBest solution:    fitness=", self.archive[0].fitness)
        print(self.archive[0])
    
        return self.archive[0]

                
    def generation(self, n=1, plot=True):
        """
        Runs n generations and plots the archive.  
        
        :return: self
        """
        for g in range(n):
            print('>> Generation:', g+1, '/', n, '\n')
            self.tournament_selection(plot=plot)
            print(self)
        self.get_archive()
        return self