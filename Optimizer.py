from Buffer import Buffer
from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from ApproximationModel import ValueNetwork, PolicyNetwork
from Utils import load_configs

import numpy as np
import pandas as pd


class Optimizer:
    def __init__(self):
        configs = load_configs()
        self.width = configs["width"]
        self.height = configs["height"]
        self.buffer_size = configs["buffer_size"]

        self.buffer = Buffer(self.buffer_size)
        self.env = BlockRelocation(self.height, self.width)
        self.model = ValueNetwork(height=self.height + 2, width=self.width)
        self.policy_network = PolicyNetwork(height=self.height + 2, width=self.width)
        self.tree_searcher = TreeSearch(self.model, BlockRelocation(self.height, self.width), self.policy_network)

    def create_training_example(self, permutations=False):
        self.env.matrix = self.env.create_instance(self.height, self.width)
        path = self.tree_searcher.find_path(self.env.matrix.copy(), search_depth=4)

        # In case the solver can't solve it with the given depth, this function is called again
        if path is None:
            print("Could not solve")
            return self.create_training_example(permutations=permutations)

        data = self.tree_searcher.move_along_path(self.env.matrix.copy(), path)

        if permutations:
            data = self.create_permutations(data)
            #print(data.shape)

        return data, len(path)

    def reinforce(self, iterations=10):
        for x in range(iterations):
            print("Iteration " + str(x))
            self.train_on_new_instances(10)
            old_sample = self.buffer.get_sample(4000, remove=True)
            self.model.train_df(old_sample)
            self.policy_network.train_df(old_sample)

    def train_on_new_instances(self, num=1):
        data_list = []
        step_list = []

        for ii in range(num):
            data, steps = self.create_training_example(permutations=True)
            data_list.append(data)
            step_list.append(steps)
            print("solved " + str(ii+1))

        data = pd.concat(data_list, ignore_index=True, sort=False)
        mean_steps = np.mean(step_list)

        train_data = data.sample(int(data.shape[0]/5))  # Hardcoded
        data = data.drop(train_data.index)

        self.model.train_df(data)
        self.policy_network.train_df(data)
        self.model.write_steps_summary(mean_steps)

        self.buffer.append(data)

    def create_permutations(self, df):
        df_list = []
        for i, row in df.iterrows():
            # creating representations
            rep = self.env.all_permutations_state(row.StateRepresentation)
            rep = [x.transpose().flatten() / 100 for x in rep]

            # creating value column
            val = [np.array(row.Value) for _ in range(len(rep))]

            # creating move and move_encoded columns
            moves = self.env.all_permutations_move(*row.Move)
            encoded = [self.tree_searcher.move_to_hot_one_encoding(m) for m in moves]

            # creating the DataFrame
            temp_df = pd.DataFrame({"StateRepresentation": rep, "Value": val, "Moves": moves, "MovesEncoded": encoded})

            # removing duplicates
            temp_df["hashable_state"] = temp_df.StateRepresentation.apply(lambda x: x.tostring())
            temp_df = temp_df.drop_duplicates(subset="hashable_state")
            temp_df = temp_df.drop(columns="hashable_state")

            df_list.append(temp_df)

        final_df = pd.concat(df_list, ignore_index=True)
        return final_df

    def warm_start(self):
        examples = self.tree_searcher.generate_basic_starting_data(num_examples=500)

        X = examples.StateRepresentation.values
        X = np.array([x.transpose().flatten() / 100 for x in X])

        y = examples.Value
        y = np.array([np.array([val], dtype=float) for val in y])

        self.model.train(X, y)

        examples_eval = self.tree_searcher.generate_basic_starting_data(num_examples=50)
        X_eval = examples_eval.StateRepresentation.values
        X_eval = np.array([x.transpose().flatten() / 100 for x in X_eval])

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
        print(X[-3].reshape(4, 6).transpose())
        # print(examples.Value)
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

    def start_network(self):
        for x in range(20):
            self.warm_start()


if __name__ == "__main__":
    test = Optimizer()
    #test.warm_start()
    #test.create_training_example(permutations=True)
    #test.train_on_new_instances(3)
    #test.warm_start()
    #test.start_network()
    test.reinforce(5)