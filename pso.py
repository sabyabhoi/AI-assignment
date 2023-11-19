import numpy as np


class ParticleSwarmOptimization:
    def __init__(self, n, c1, c2, fitness) -> None:
        self.n = n  # number of particles
        self.c1 = c1
        self.c2 = c2
        self.fitness = fitness

    def initialize(self):
        self.x = np.random.randint(1, [11, 257, 129], size=(self.n, 3))
        self.v = np.zeros((self.n, 3))

    def repair(self, x):
        x = (np.rint(x)).astype(int)
        x[0] = min(10, max(1, x[0]))
        x[1] = min(256, max(1, x[1]))
        x[2] = min(128, max(1, x[2]))
        return x

    def run(self, epochs=50):
        print("here")
        self.initialize()
        self.lbest = self.x
        self.gbest = self.lbest[0]

        for iter in range(epochs):
            print("Iteration", iter + 1)

            for i in range(self.n):
                print("Individual", i + 1, end=" : ")
                print(self.x[i])

                if self.fitness(self.x[i]) < self.fitness(self.lbest[i]):
                    self.lbest[i] = self.x[i]

                print("Local Best of individual", i + 1, end=" : ")
                print(self.lbest[i])

                if self.fitness(self.lbest[i]) < self.fitness(self.gbest):
                    self.gbest = self.lbest[i]

                print("Global Best : ", self.gbest)

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
                print("Individual", i + 1, "after repair :", self.x[i])
        return self.gbest
