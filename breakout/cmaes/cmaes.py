class CMAES:
    def __init__(self, n, create, fitness, mutate, breed):
        '''
        CMAES class constructor
        Args:
            n: number of individuals to create
            create: function that returns a randomized individual
            fitness: function that returns a fitness given an individual
            mutate: function that mutates
            breed:
        '''
        self.n = n
        self.create = create
        self.fitness = fitness
        self.mutate = mutate
        self.breed = breed

    def sample(self):
        '''
        If the
        Returns: a new individual sampled from

        '''