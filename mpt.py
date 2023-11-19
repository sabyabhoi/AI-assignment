from simulated_annealing import SimulatedAnnealing
import numpy as np

sigma = np.array([[0.1693, 0.0460, 0.0043, 0.0068, 0.0090, 0.0053],
                  [0.0460, 0.2872, 0.0083, 0.0671, 0.0175, 0.0354],
                  [0.0043, 0.0083, 0.0343, 0.0024, 0.0028, 0.0043],
                  [0.0068, 0.0671, 0.0024, 0.1943, 0.0142, 0.0510],
                  [0.0090, 0.0175, 0.0028, 0.0142, 0.0397, 0.0085],
                  [0.0053, 0.0354, 0.0043, 0.0510, 0.0085, 0.0470]])
rf = .0721
exp_ret = np.matrix('95.67; 180.59; 31.90; 12.53; 68.31; 20.03') / 100


def init_w(size, randomize=True):
    if randomize:
        w = np.random.uniform(0, 1, size)
        w = w.reshape(size, 1) / w.sum()
        return w
    else:
        w = np.zeros((size, 1))
        w[0][0] = 1
        return w


def sharpe(w: np.ndarray):
    return ((w.T @ exp_ret - rf) / np.sqrt(w.T @ sigma @ w))


def constraint(w: np.ndarray):
    return np.all(w >= 0) and np.abs(w.sum() - 1.0) <= 1e-2


if __name__ == '__main__':
    for i in range(10):
        s = SimulatedAnnealing(init_w(6),
                               10,
                               .1,
                               .6,
                               4000,
                               sharpe,
                               constraint,
                               min_state=init_w(6, randomize=False))
        print(sharpe(s.run()))
