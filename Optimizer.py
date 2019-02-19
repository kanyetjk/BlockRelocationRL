from Buffer import Buffer
from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from ApproximationModel import ValueNetwork, PolicyNetwork, CombinedModel, EstimatorWrapper
from Utils import load_configs
from sklearn.model_selection import train_test_split
from KerasModel import ValueNetworkKeras

import numpy as np
import pandas as pd
import time
import logging


class Optimizer:
    def __init__(self):
        configs = load_configs("Configs.json")
        value_configs = load_configs("Configs_ValueNN.json")
        policy_configs = load_configs("Configs_PolicyNN.json")

        logging.basicConfig(format='%(asctime)s %(message)s', filename=configs["log_name"], level=logging.WARNING,
                            datefmt="%Y-%m-%d %H:%M:%S")
        logging.info("Starting Optimizer.")

        self.width = configs["width"]
        self.height = configs["height"]
        self.buffer_size = configs["buffer_size"]

        self.buffer = Buffer(self.buffer_size)
        self.env = BlockRelocation(self.height, self.width)
        self.model = ValueNetwork(configs=value_configs)
        self.policy_network = PolicyNetwork(configs=policy_configs)
        self.combined_model = CombinedModel(configs=configs)
        self.value_wrapper = EstimatorWrapper(self.model)
        self.policy_wrapper = EstimatorWrapper(self.policy_network)

        self.tree_searcher = TreeSearch(self.value_wrapper, BlockRelocation(self.height, self.width), self.policy_wrapper)

        self.baseline_params = {"search_depth": 4,
                                "epsilon": 0.1,
                                "threshold": 0.01,
                                "drop_percent": 0.25,
                                "factor": 0.01}

        self.current_search_params = {"search_depth": 5,
                                      "epsilon": 0.1,
                                      "threshold": 0.05,
                                      "drop_percent": 0.3,
                                      "factor": 0.01}

        logging.info("Start up process complete.")

    def create_training_example(self, permutations=True, units=8):
        if units < self.height * self.width:
            matrix = self.env.create_instance_random(units)
        else:
            matrix = self.env.create_instance(self.height, self.width)

        v1 = self.current_search_params
        path = self.tree_searcher.find_path_2(matrix.copy(), **v1)
        # In case the solver can't solve it with the given depth, this function is called again
        if path is None:
            return self.create_training_example(permutations=permutations)

        data = self.tree_searcher.move_along_path(matrix.copy(), path)

        if permutations:
            data = self.create_permutations(data)
        else:
            data = self.prepare_data_for_model(data)

        return data, len(path)

    def train_on_new_instances(self, num=1, units=10):
        data_list = []
        step_list = []

        for ii in range(num):
            data, steps = self.create_training_example(permutations=True, units=units)
            data_list.append(data)
            step_list.append(steps)
            #print(steps)

        data = pd.concat(data_list, ignore_index=True, sort=False)
        with open('up_to_8.csv', 'a') as f:
            data.to_csv(f, header=False, index=False)
        mean_steps = np.mean(step_list)

        train_data = data.sample(int(data.shape[0] / 3))  # Hardcoded

        self.model.train_df(train_data)
        self.policy_network.train_df(train_data)
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

    def test_networks(self):
        matrix = self.env.create_instance_random(12)
        while self.env.can_remove_matrix(matrix):
            matrix = self.env.remove_container_from_matrix(matrix)
        print(matrix)
        m = matrix.transpose().flatten()
        m = np.array([np.array(m) / 16])
        print(m)
        print(list(self.model.predict(m)))
        print(list(self.policy_network.predict(m)))

    def training_on_csv(self):
        data = pd.read_csv("up_to_8.csv")
        print(data.shape)
        data["StateRepresentation"] = data["StateRepresentation"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
        data["MovesEncoded"] = data["MovesEncoded"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))

        data["hashed"] = data["StateRepresentation"].apply(lambda s: s.tostring())
        data = data.drop_duplicates(subset="hashed")
        data = data.drop(columns=["hashed"])
        data = data.reset_index(drop=True)
        print(data.shape)
        train_data, test_data = train_test_split(data, shuffle=True, test_size=0.25)

        for ii in range(20):
            #self.model.train_df(train_data)
            #self.model.evaluate_df(test_data)
            self.policy_network.train_df(train_data)
            self.policy_network.evaluate_df(test_data)
            #self.combined_model.train_df(train_data)
            #self.combined_model.evaluate_df(test_data)
            print(str(ii) + " done!")

    def evaluate_parameters(self):
        v1 = {"search_depth": 4, "epsilon": 0.075, "threshold": 0.07, "drop_percent": 0.7, "factor": 0.15}
        v2 = {"search_depth": 4, "epsilon": 0.1, "threshold": 0.07, "drop_percent": 0.6, "factor": 0.12}
        v3 = {"search_depth": 4, "epsilon": 0.1, "threshold": 0.05, "drop_percent": 0.3, "factor": 0.1}
        v4 = {"search_depth": 5, "epsilon": 0.1, "threshold": 0.05, "drop_percent": 0.3, "factor": 0.01}

        version_list = [v1, v2, v3, v4]
        results = [[] for _ in version_list]
        times = [[] for _ in version_list]

        for ii in range(10):
            print(ii)
            matrix = self.env.create_instance_random(14)
            for i, version in enumerate(version_list):
                print("___" + str(i))
                start = time.time()
                m = len(self.tree_searcher.find_path_2(matrix, **version))
                end = time.time()
                results[i].append(m)
                times[i].append(end-start)

        for i in range(len(results)):
            print(results[i])
            print(np.mean(times[i]))

    def find_best_parameters(self):
        matrices = [self.env.create_instance_random(10) for _ in range(10)]
        baseline_moves, baseline_seconds = self.evaluate_params(matrices, self.baseline_params)
        print(baseline_moves, baseline_seconds)
        new_params = self.baseline_params.copy()

        for _ in range(25):
            new_params["drop_percent"] *= 1.075
            new_params["factor"] *= 1.015
            new_params["threshold"] += 0.003
            print(new_params)
            current_moves, current_seconds = self.evaluate_params(matrices, new_params)
            print(current_moves, current_seconds)

    def evaluate_params(self, matrices_input, params):
        moves = 0
        start = time.time()
        for m in matrices_input:
            start = time.time()
            m = len(self.tree_searcher.find_path_2(m, **params))
            moves += m
        end = time.time()
        seconds = end-start
        return moves, seconds

    def test_combinded_model(self):
        """
        matrix = self.env.create_instance_random(10)
        m = matrix.transpose().flatten()
        m = np.array([np.array(m) / 16])
        a = self.combined_model.predict(m)
        a = list(a)"""

        a, moves = self.create_training_example()
        self.combined_model.train_df(a)

    def test_wrapper(self):
        a, moves = self.create_training_example()
        start = time.time()
        for _ in range(1):
            print(a.StateRepresentation[0])
            print(list(self.value_wrapper.predict_df(a)))
        end = time.time()
        print(end-start)
        start = time.time()
        for _ in range(1):
            print(list(self.model.predict_df(a)))
        end = time.time()
        print(end-start)

    def reinforce(self, iterations=20, units=12):
        for x in range(iterations):
            start = time.time()
            print("Iteration " + str(x))
            self.train_on_new_instances(25, units=units)
            old_sample = self.buffer.get_sample(2000, remove=True)
            self.model.train_df(old_sample)
            self.policy_network.train_df(old_sample)
            end = time.time()
            #print(end-start)

    def train_and_update_models(self):
        data = self.buffer.get_sample(size=self.buffer.max_size)
        self.policy_network.train_df(data)
        self.model.train_df(data)
        self.value_wrapper = EstimatorWrapper(self.model)
        self.policy_wrapper = EstimatorWrapper(self.policy_network)
        self.tree_searcher = TreeSearch(self.value_wrapper, BlockRelocation(self.height, self.width),
                                        self.policy_wrapper)

    def full_experiment(self):
        """
        for ii in range(8, 13):
            logging.info(f"Training: Currently training on {ii} units.")
            self.reinforce(iterations=10, units=ii)
            self.train_and_update_models()
        """
        performance_test_matrices = [self.env.create_instance(self.height, self.width) for _ in range(20)]
        test_params = {"search_depth": 5, "epsilon": 0.1, "threshold": 0.05, "drop_percent": 0.4, "factor": 0.015}

        for ii in range(13, 17):
            print(f"Currently training on: {ii} units.")
            logging.info(f"Training: Currently training on {ii} units.")
            moves, seconds = self.evaluate_params(performance_test_matrices, test_params)
            print(f"Moves: {moves}, seconds: {round(seconds, 2)}")
            logging.info(f"Evaluation: Total Moves: {moves}, Total Seconds: {seconds}")
            # find params
            self.reinforce(iterations=1, units=ii)
            self.train_and_update_models()

        self.buffer.storage.to_csv("4x4_data_first_half.csv", index=False)
        self.buffer = Buffer(self.buffer_size)

        for ii in range(10):
            self.reinforce(iterations=10, units=self.height*self.width)
            print(f"Training with all units. Currently on iteration {ii}.")
            moves, seconds = self.evaluate_params(performance_test_matrices, test_params)
            print(f"Moves: {moves}, seconds: {round(seconds, 2)}")
            print(moves, seconds)

        # find best parameters
        # run experiment on test instances

    def test_stupid_wrapper(self):
        a, moves = self.create_training_example()
        for _ in range(2):
            print(list(self.value_wrapper.predict_df(a)))
        value_configs = load_configs("Configs_ValueNN.json")
        self.model = ValueNetwork(value_configs)
        self.value_wrapper = EstimatorWrapper(self.model)
        print(list(self.value_wrapper.predict_df(a)))

    def test_keras(self):
        data, steps = self.create_training_example(permutations=True, units=10)
        X = data.StateRepresentation.values / 16
        X = np.array([x for x in X])

        y = data.Value.values
        y = np.array([np.array([val], dtype=float) for val in y])
        self.keras_model = ValueNetworkKeras()
        self.keras_model.train(X, y)

if __name__ == "__main__":
    test = Optimizer()
    #test.test_combinded_model()
    #test.reinforce(10)
    #test.test_value_network()
    #test.train_on_new_instances(1)
    #test.training_on_csv()
    # learning to learn better than your teacher
    # test.create_training_example(permutations=False, units=14)
    #test.evaluate_parameters()
    #test.test_wrapper()
    #test.find_best_parameters()
    #test.full_experiment()
    #test.test_stupid_wrapper()
    test.test_keras()
