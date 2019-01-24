from Buffer import Buffer
from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from ApproximationModel import ValueNetwork
import numpy as np


class Optimizer:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height

        self.buffer = Buffer(size=200000)
        self.env = BlockRelocation(self.height, self.width)
        self.model = ValueNetwork(height=self.height+2, width=self.width)
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

    def warm_start(self):
        # Here Multiprocessing usefull?
        examples = self.tree_searcher.generate_basic_starting_data(num_examples=5000)
        X = examples.StateRepresentation.values
        X = np.array([x.transpose().flatten()/100 for x in X])

        y = examples.Value
        y = np.array([np.array([val], dtype=float) for val in y])
        self.model.train(X, y)
        #print(X)
        #print(y)
        print(X.shape)
        print(self.model.evaluate(X, y))

    def compare_model(self):
        examples = self.tree_searcher.generate_basic_starting_data(num_examples=10)
        X = examples.StateRepresentation.values
        X = np.array([x.transpose().flatten() / 100 for x in X])

        y = examples.Value
        y = np.array([np.array([val], dtype=float) for val in y])
        print(self.model.evaluate(X, y))

        p = list(self.model.predict(X))
        examples["predicted"] = p
        print(X[-3].reshape(4,6).transpose())
        #print(examples.Value)
        print(examples[["Value", "predicted"]])

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


test = Optimizer(4,4)
test.compare_model()