class AFPPopulation(Population):
    """
    AFPPopulation is based on the Population class but implements Age-Fitness Pareto Survival instead of Tournament selection.
    """
    def __init__(self, variables=None, data=None, init_size=10, pop_size=100):
        super().__init__(variables=variables, data=data, init_size=init_size, pop_size=pop_size)
        
        # The Program objects of the population are replaced with AFP_Program objects to enable custom genetic operators.
        self.programs = [AFP_Program(variables=self.variables, model=clone(self.model)).generate(init_size).get_features() for _ in range(pop_size)]
        self.size = init_size
        
    def __repr__(self):
        """
        Description of the population 
        
        :return: String
        """
        return 'AFP' + super().__repr__()
    
    def plot_fitness(self):
        """
        Plots a scatter plot (age, fitness) with colors related to the 'Age-Fitness Pareto fitness'.
        
        :return: self
        """
        fig = plt.figure(figsize=(8, 8))
        a = [p.age for p in self.programs]
        f = [p.fitness for p in self.programs]
        afp_f = [p.AFP_fitness for p in self.programs]
        sc = plt.scatter(a, f, c=afp_f)
        plt.xlabel('Age')
        plt.ylabel('Fitness')
        plt.ylim(0, 1)
        cbar = plt.colorbar(sc)
        sc.set_clim(0.0, 1.0)
        cbar.ax.set_ylabel('Age-Fitness Pareto fitness', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15
        plt.show()
        print('Mean', np.mean(f), ' | Std', np.std(f), ' | Max', np.max(f), '\n')
        return self
    
    
   
    def age_fitness_selection(self, verbose=False, plot=True):
        """
        Implements Age-Fitness Pareto Survival
        
        :return: self
        """
        pop_size = len(self.programs)
        
        # Increment age
        for p in self.programs:
            p.age += 1
        
        # Random parent selection
        def get_parent():
            i = rd.randint(0, len(self.programs)-1)
            return self.programs[i]
        
        # New population
        new_pop = []
        i_max = pop_size - 1  # Max index in self.programs

        # Number of new programs from crossover
        n_cross = math.floor(pop_size * self.params['crossover'])
        # Number of new programs from mutation
        n_mut = math.floor(pop_size * self.params['mutation'])
        # Number of new programs from reproduction
        n_new = pop_size - n_cross - n_mut
        
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
        
        # Addition of new random programs with age 0
        for i in range(n_new):
            new = AFP_Program(variables=self.variables, model=clone(self.model), age=0).generate(self.size)
            new_pop.append(new)
           
        # Concatenation of the old and new populations: parents, children and new solutions compete to be in the new population
        self.programs = new_pop + self.programs
        
        
        def update_fitness(pop, ntot=None):
            """
            Computes the AFP_fitness for each solution in the population.
            It is called recusively on the population without the previous Pareto front while it is not empty.

            :return: list of solutions on the first Pareto front
            """
            # If the population is empty, return an empty list
            if len(pop)==0:
                return []
            
            # Keep the size of the global population constant through the recursive calls
            if ntot is None:
                ntot=len(pop)
            
            pareto, non_pareto = [], []
            
            for p in pop:
                dominated = sum([a.AFP_dominates(p) for a in pop]) # Number of other solutions dominating p
                if dominated == 0:
                    pareto.append(p) # p is on the Pareto front
                else:
                    non_pareto.append(p) # p is not on the Pareto front
            
            # Compute the AFP_fitness from age and fitness
            for p in pareto:
                p.AFP_fitness = len(non_pareto)/(ntot)
                
            update_fitness(non_pareto, ntot=ntot)
            
            return pareto
        
        # Update the fitness
        self.evaluation(verbose=verbose, plot=False)
        
        # Update the AFP_fitness
        update_fitness(self.programs)
        
        # Sort the population with AFP_fitness
        self.programs.sort(key=lambda x:x.AFP_fitness, reverse=True)
        
        # Select the best solutions to create the new population of same size as the previous ones.
        self.programs = self.programs[:pop_size]
        
        if plot: self.plot_fitness()
        return self
    
        
    def generation(self, n=1, plot=True):
        """
        Runs n generations and plots the archive.  
        
        :return: self
        """
        for g in range(n):
            print('>> Generation:', g+1, '/', n, '\n')
            self.age_fitness_selection(plot=plot)
            print(self)
        self.get_archive()
        return self