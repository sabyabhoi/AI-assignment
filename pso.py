import numpy as np


class ParticleSwarmOptimization:
    def __init__(self, n, c1, c2, fitness) -> None:
        self.n = n  # number of particles
        self.c1 = c1
        self.c2 = c2
        self.fitness = fitness

    def initialize(self):
        self.x = np.random.randint(1, [8, 257, 129], size=(self.n,3))
        self.v = np.zeros((self.n, 3))

    def repair(self, x):
        x = (np.rint(x)).astype(int)
        x[0] = min(7, max(1, x[0]))
        x[1] = min(256, max(1, x[1]))
        x[2] = min(128, max(1, x[2]))
        return x

    def run(self, epochs=50):
        self.initialize()    
        self.lbest = np.copy(self.x)
        self.gbest = np.copy(self.lbest[0])

        for iter in range(epochs):
            print("\nIteration", iter+1)

            for i in range(self.n):
                if self.fitness(self.x[i]) < self.fitness(self.lbest[i]):
                    print("Local best changed for particle", i+1)
                    self.lbest[i] = np.copy(self.x[i])
                
                if self.fitness(self.lbest[i]) < self.fitness(self.gbest):
                    print("Global best changed")
                    self.gbest = np.copy(self.lbest[i])

                print("Current fitness :", self.fitness(self.x[i]),", Local best fitness :", self.fitness(self.lbest[i]), ", Global best fitness :", self.fitness(self.gbest))
      
            for i in range(self.n):
                r1 = np.random.rand(3)
                r2 = np.random.rand(3)
                self.v[i] = (
                    self.v[i]
                    + self.c1 * np.multiply(r1, (self.lbest[i] - self.x[i]))
                    + self.c2 * np.multiply(r2, (self.gbest - self.x[i]))
                )
                print("Velocity of individual", i + 1, ":", self.v[i])
                self.x[i] = self.repair(self.x[i] + self.v[i])
        
        return self.gbest
