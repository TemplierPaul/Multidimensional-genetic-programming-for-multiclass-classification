class LexPopulation(Population):
    """
    LexPopulation is based on the Population class but implements Lexicase Selection instead of Tournament selection.
    """
    def __init__(self, variables=None, data=None, init_size=10, pop_size=100):
        super().__init__(variables=variables, data=data, init_size=init_size, pop_size=pop_size)
        
    def __repr__(self):
        return 'Lex' + super().__repr__()
    
    def lexicase_selection(self, verbose=False, plot=True):
        """
        Implements Lexicase Selection, Mutation, Crossover, and Reproduction steps of the Genetic Algorithm.  
        
        :return: self
        """
        # Update the fitness
        self.evaluation(verbose=verbose, plot=plot)
        
        # Get the list of correct predictions
        df = pd.DataFrame(self.dataset['data'], columns=self.dataset['feature_names'])
        for i in range(len(self.programs)):
            p = self.programs[i]
            df_p = p.compute(df).df # transform data into the new features space
            X_train_p, X_test_p, y_train, y_test = train_test_split(df_p,
                                                            self.dataset['target'],
                                                            test_size=self.params['tournament']['test_size'],
                                                            random_state=42)
            
            
            
            if np.size(X_train_p, 1) == 0:
                p.lexicase_eval = [False for i in range(len(y_test))]
            else:
                pred = p.model.predict(X_test_p).tolist() # Predict the class of the test data points
                y_test = y_test.tolist()
                p.lexicase_eval = [y_test[i] == pred[i] for i in range(len(y_test))]# Boolean list: True if correctly predicted, else False
        
        # Lexicase selection for the parent selection
        def get_parent():
            # Take a randomly ordered list of points to evaluate the solutions
            i_list = list(range(len(y_test)))
            rd.shuffle(i_list)
            
            pop = self.programs
            for i in i_list: 
                new_pop = []
                for p in pop:
                    if p.lexicase_eval[i]: # p stays in the next population if it correctly predicted the ith item, i being random
                        new_pop.append(p)
                if len(new_pop)>0:
                    pop = new_pop # Keep iterating while there are surviving solutions in new_pop
                else:
                    if len(pop) == 1: # If only one was correct, select it
                        return pop[0]
                    else: # If there are multiple solution with equal performance, return one at random
                        return rd.choice(pop)
            return rd.choice(pop) # If some solutions got a perfect score, return one of them
        
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
        
        # Crossover
        if verbose: print('Crossover')
        for k in range(n_cross):
            p1, p2 = get_parent(), get_parent()

            # Add the children of p1 and p2 to the new population
            c1, c2 = p1.crossover(p2)
            new_pop.append(c1)

        # Mutation
        if verbose: print('Mutation')
        for k in range(n_mut):
            # Tournament for initial program to be mutated
            p = get_parent().copy()

            # Mutate a copy of the selected program and add it to the new population
            p.mutate(add_len=int(len(self.programs)) / 10)
            new_pop.append(p)
        
        # Reproduction: the best one stay in the new population
        if n_reprod > 0:
            if verbose: print('Reproduction:', n_reprod)
            # Copy the best fit elements into the new programs list
            sorted_progs = sorted(self.programs, key=lambda x: x.fitness, reverse=True)
            for k in range(n_reprod):
                new_pop.append(sorted_progs[k])

        self.programs = new_pop
        return self
    
    def generation(self, n=1, plot=True):
        """
        Runs n generations and plots the archive.  
        
        :return: self
        """
        for g in range(n):
            print('>> Generation:', g+1, '/', n, '\n')
            self.lexicase_selection(plot=plot)
            print(self)
        self.get_archive()
        return self