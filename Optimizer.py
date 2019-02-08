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

    def create_training_example(self, permutations=True):
        matrix = self.env.create_instance_random(8)
        path = self.tree_searcher.find_path_2(matrix.copy(), search_depth=4)

        # In case the solver can't solve it with the given depth, this function is called again
        if path is None:
            return self.create_training_example(permutations=permutations)

        data = self.tree_searcher.move_along_path(matrix.copy(), path)

        if permutations:
            data = self.create_permutations(data)
        else:
            data = self.prepare_data_for_model(data)

        return data, len(path)

    def reinforce(self, iterations=20):
        for x in range(iterations):
            print("Iteration " + str(x))
            self.train_on_new_instances(30)
            old_sample = self.buffer.get_sample(750, remove=True)
            self.model.train_df(old_sample)
            self.policy_network.train_df(old_sample)

    def train_on_new_instances(self, num=1):
        data_list = []
        step_list = []

        for ii in range(num):
            data, steps = self.create_training_example(permutations=True)
            data_list.append(data)
            step_list.append(steps)

        data = pd.concat(data_list, ignore_index=True, sort=False)
        mean_steps = np.mean(step_list)

        train_data = data.sample(int(data.shape[0]/4))  # Hardcoded
        data = data.drop(train_data.index)

        # return
        self.model.train_df(data)
        self.policy_network.train_df(data)
        self.model.write_steps_summary(mean_steps)

        self.buffer.append(data)

    def prepare_data_for_model(self, data):
        # TODO DONT WANT TO CHANGE THE COLUMN NAME HERE
        data["StateRepresentation"] = data["StateRepresentation"].apply(lambda x: x.transpose().flatten())
        data.columns = ["Moves", "StateRepresentation", "Value"]
        data["MovesEncoded"] = data["Moves"].copy()
        data["MovesEncoded"] = data["MovesEncoded"].apply(lambda x: self.tree_searcher.move_to_hot_one_encoding(x))
        return data

    def create_permutations(self, df):
        df_list = []
        for i, row in df.iterrows():
            # creating representations
            rep = self.env.all_permutations_state(row.StateRepresentation)
            rep = list(rep)

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

    def test_value_network(self):
        matrix = self.env.create_instance_random(8)
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)
        print(matrix)
        m = matrix.transpose().flatten()
        m = np.array([np.array(m) / 16])
        print(m)
        print(list(self.model.predict(m)))
        print(list(self.policy_network.predict(m)))

    def test_permutations(self):
        a = self.create_training_example(permutations=False)



if __name__ == "__main__":
    test = Optimizer()
    #test.reinforce(20)
    test.test_value_network()
    #test.train_on_new_instances(1)
