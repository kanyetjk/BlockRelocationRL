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

    def create_training_example(self, permutations=False):
        # TODO ADD MOVES
        # TODO Needs to stop at some number of moves because it doesn't always solve the problem
        self.env.create_instance(self.height, self.width)
        path = self.tree_searcher.find_path(self.env.matrix.copy(), search_depth=4)
        data = self.tree_searcher.move_along_path(self.env.matrix.copy(), path)

        # In case the solver can't solve it with the given depth
        if data is None:
            return self.create_training_example(permutations=permutations)

        print(data)

        if permutations:
            # TODO create permutations
            pass

        self.model.train_df(data)
        return data

    def warm_start(self):
        examples = self.tree_searcher.generate_basic_starting_data(num_examples=500)
        X = examples.StateRepresentation.values
        X = np.array([x.transpose().flatten()/100 for x in X])

        y = examples.Value
        y = np.array([np.array([val], dtype=float) for val in y])
        self.model.train(X, y)

        examples_eval = self.tree_searcher.generate_basic_starting_data(num_examples=50)
        X_eval = examples_eval.StateRepresentation.values
        X_eval = np.array([x.transpose().flatten()/100 for x in X_eval])

        y_eval = examples_eval.Value
        y_eval = np.array([np.array([val], dtype=float) for val in y_eval])

        print(self.model.evaluate(X_eval, y_eval))

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
#a = test.tree_searcher.find_path(test.env.create_instance(4,4), search_depth=4)
#test.compare_model()
test.create_training_example(permutations=False)
#print(a)