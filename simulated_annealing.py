import random
import numpy as np
import math


class SimulatedAnnealing:

    def __init__(self, start_state: np.ndarray, T, Tmin, k, n, f, constraint,
                 min_state: np.ndarray):
        self.state = start_state
        self.T = T
        self.Tmin = Tmin
        self.k = k
        self.n = n
        self.f = f
        self.constraint = constraint
        self.min_state = min_state
        pass

    def neighbour(self, state):
        n = len(state)
        return (state + np.random.normal(0, 1, n).reshape(n, 1)) / n

    def run(self):
        while self.T > self.Tmin:
            for _ in range(self.n):
                if self.f(self.state) > self.f(self.min_state):
                    self.min_state = self.state

                new_state = self.neighbour(self.state)
                change_in_energy = self.f(new_state) - self.f(self.state)
                exp_term = math.exp(change_in_energy / self.T)

                if self.constraint(new_state) and (change_in_energy > 0
                                                   or exp_term
                                                   >= random.uniform(0, 1)):
                    self.state = new_state

            self.T *= self.k
        return self.min_state
