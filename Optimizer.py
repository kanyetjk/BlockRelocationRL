from Buffer import Buffer
from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from ApproximationModel import ApproximationModel


class Optimizer:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height

        self.buffer = Buffer(size=200000)
        self.env = BlockRelocation(self.height, self.width)
        self.model = ApproximationModel()
        self.tree_searcher = TreeSearch(self.model, BlockRelocation(self.height, self.width))

    def create_training_example(self, rows=3, permutations=False):
        air = self.height + 2 - rows
        self.env.create_instance(rows, self.width, air)
        path = self.tree_searcher.find_path(self.env.matrix.copy(), 3)
        data = self.tree_searcher.move_along_path(self.env.matrix.copy(), path)

        if permutations:
            # TODO create permutations
            pass
        return data


    def train(self, next_examples=20, search_depth=3):
        # create training example
        # train on that example
        # add to buffer
        # run batch from buffer
        pass

    def find_optimal_solution_bfs(self, matrix):
        pass

    def find_fast_solution(self, matrix):
        pass
