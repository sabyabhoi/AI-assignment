import random
import numpy as np
import math


class SimulatedAnnealing:
    def __init__(
        self,
        start_state: np.ndarray,
        T,
        Tmin,
        k,
        n: int,
        f,
        constraint,
        max_state: np.ndarray,
    ):
        self.state = start_state
        self.T = T
        self.Tmin = Tmin
        self.k = k  # temperature reduction factor
        self.n = n  # no. of iterations
        self.f = f  # fitness function
        self.constraint = constraint
        self.max_state = max_state

    def neighbour(self, state):
        n = len(state)
        x, y, z = state
        x += np.random.randint(-2, 2)
        x = min(x, 10)
        x = max(x, 1)
        y += np.random.randint(-32, 32)
        y = min(y, 256)
        y = max(y, 1)
        z += np.random.randint(-16, 16)
        z = min(z, 128)
        z = max(z, 1)
        return (x, y, z)

    def run(self):
        while self.T > self.Tmin:
            for i in range(self.n):
                print(f"[TEMP = {self.T}] {i + 1}: ")
                if self.f(self.state) > self.f(self.max_state):
                    self.max_state = self.state

                new_state = self.neighbour(self.state)

                change_in_energy = self.f(new_state) - self.f(self.state)
                exp_term = math.exp(change_in_energy / self.T)

                if change_in_energy > 0 or exp_term >= random.uniform(0, 1):
                    self.state = new_state

            self.T *= self.k
        return self.max_state
