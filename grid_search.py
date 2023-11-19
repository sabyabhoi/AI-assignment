class GridSearch:
    def __init__(self, init_state: (int, int, int), f, search_space: list[list[int]]):
        self.init_state = init_state
        self.f = f
        self.search_space = search_space

    def run(self):
        best_state = self.init_state
        for win_size in self.search_space[0]:
            for hidden1 in self.search_space[1]:
                for hidden2 in self.search_space[2]:
                    if self.f(best_state) < self.f((win_size, hidden1, hidden2)):
                        best_state = (win_size, hidden1, hidden2)

        return best_state
