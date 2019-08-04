from Buffer import Buffer
from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from ApproximationModel import ValueNetwork, PolicyNetwork, CombinedModel, EstimatorWrapper
from Utils import load_configs, load_obj, save_obj
from sklearn.model_selection import train_test_split
from KerasModel import ValueNetworkKeras, PolicyNetworkKeras
from Benchmark import Benchmark


import numpy as np
import pandas as pd
import time
import os.path
#import logging


class Optimizer:
    def __init__(self):
        configs = load_configs("Configs.json")
        value_configs = load_configs("Configs_ValueNN.json")
        policy_configs = load_configs("Configs_PolicyNN.json")

        #logging.basicConfig(format='%(asctime)s %(message)s', filename=configs["log_name"], level=logging.WARNING,
                            #datefmt="%Y-%m-%d %H:%M:%S")
        #logging.info("Starting Optimizer.")

        self.width = configs["width"]
        self.height = configs["height"]
        self.buffer_size = configs["buffer_size"]

        self.filename = configs["data_filename"]
        self.experiment_name = "Deviations/" + configs["val_filename"]

        self.buffer = Buffer(self.buffer_size)
        self.env = BlockRelocation(self.height, self.width)
        #self.model = ValueNetwork(configs=value_configs)
        #self.policy_network = PolicyNetwork(configs=policy_configs)
        #self.combined_model = CombinedModel(configs=configs)
        #self.value_wrapper = EstimatorWrapper(self.model)
        #self.policy_wrapper = EstimatorWrapper(self.policy_network)
        self.value_net = ValueNetworkKeras(value_configs)
        self.policy_net = PolicyNetworkKeras(policy_configs)

        self.tree_searcher = TreeSearch(self.value_net, BlockRelocation(self.height, self.width), self.policy_net)
        self.tree_searcher.std_vals = load_obj(self.experiment_name)

        self.baseline_params = {"search_depth": 5,
                                "epsilon": 0.1,
                                "threshold": 0.01,
                                "drop_percent": 0.25,
                                "factor": 0.01}

        self.current_search_params = {"search_depth": 5,
                                      "epsilon": 0.1,
                                      "threshold": 0.05,
                                      "drop_percent": 0.3,
                                      "factor": 0.05}

        self.dfs_params_hq = {"stop_param": 4, "k": 12}
        self.dfs_params_fast = {"stop_param": 1, "k": 12}

        #logging.info("Start up process complete.")

    def create_training_example(self, permutations=True, units=8):
        if units < self.height * self.width:
            matrix = self.env.create_instance_random(units)
        else:
            matrix = self.env.create_instance(self.height, self.width)

        if units < 6:
            path = self.tree_searcher.find_path_2(matrix)
        else:
            path = self.tree_searcher.find_path_dfs(matrix.copy())

        # In case the solver can't solve it with the given depth, this function is called again
        if not path:
            return self.create_training_example(permutations=permutations, units=units)

        try:
            data = self.tree_searcher.move_along_path(matrix.copy(), path)
        except TypeError:
            return self.create_training_example(permutations=permutations, units=units)

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

        data = pd.concat(data_list, ignore_index=True, sort=False)

        self.calculate_deviations(data)

        with open(self.filename, 'a') as f:
            data.to_csv(f, header=False, index=False)

        train_data = data.sample(int(data.shape[0] / 3))
        self.value_net.train_df(train_data, epochs=1, validation=False)
        self.policy_net.train_df(train_data, epochs=1, validation=False)

        self.buffer.append(data)

    def calculate_deviations(self, data):
        data = data.copy()
        data["pred"] = list(self.value_net.predict_df(data))
        data["pred"] = data["pred"].apply(lambda x: x[0])
        data["Value"] = data["Value"].astype('int64')
        data["deviation"] = abs(data["pred"] - data["Value"])

        deviations = data.groupby("Value")["deviation"].mean()

        new_vals = deviations.to_dict()
        old_vals = self.tree_searcher.std_vals

        for key in new_vals.keys():
            if key not in old_vals:
                old_vals[key] = new_vals[key]
            else:
                old_vals[key] = (old_vals[key] + new_vals[key]) / 2

        self.tree_searcher.std_vals = old_vals
        save_obj(old_vals, self.experiment_name)
        return old_vals

    def test_deviations(self):
        for i in range(10):
            data, steps = self.create_training_example(units=11)
            print(steps)
        print("DONE")
        #print(self.calculate_deviations(data))


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

    def reinforce(self, iterations=20, units=12, instances=50):
        print("Starting reinfoce with {} iterations and {} units.".format(iterations, units))
        for x in range(iterations):
            start = time.time()
            print("Iteration " + str(x+1))
            self.train_on_new_instances(instances, units=units)

            end = time.time()
            print(end-start)

    def train_and_update_models(self, epochs=3):
        data = self.buffer.get_sample(size=self.buffer.max_size)
        print("Training Policy Network ...")
        self.policy_net.train_df(data, epochs=epochs, validation=False)
        print("Training Value Network ...")
        self.value_net.train_df(data, epochs=epochs, validation=False)

    def train_on_csv(self, filename):
        data = pd.read_csv(filename)
        print(data.shape)
        data["StateRepresentation"] = data["StateRepresentation"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
        data["MovesEncoded"] = data["MovesEncoded"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))

        data["hashed"] = data["StateRepresentation"].apply(lambda s: s.tostring())
        data = data.drop_duplicates(subset="hashed")
        data = data.drop(columns=["hashed"])
        data = data.reset_index(drop=True)
        print(data.shape)
        train_data, test_data = train_test_split(data, shuffle=True, test_size=0.1)

        for i in range(5):
            print("Currently on run {} of training.".format(i+1))
            self.policy_net.train_df(train_data)
            self.value_net.train_df(train_data)

            print("Policy Network Statistics:")
            print(self.value_net.eval(test_data))

            print("Value Network Statistics:")
            print(self.policy_net.eval(test_data))
        print("Training finished!")

    def full_experiment(self):
        total_container = self.width * self.height
        for ii in range(5, total_container-5):
            print("Training: Currently training on {} units.".format(ii))
            self.reinforce(iterations=10, units=ii)
            if ii % 3 == 0:
                data = self.buffer.get_sample(self.buffer.size, remove=True)
                self.policy_net.retrain_model(data)
                self.value_net.retrain_model(data)
                self.buffer.increase_max_size(0.1)
            self.train_and_update_models()

        for ii in range(total_container-5, total_container+1):
            print("Currently training on: {ii} units.")
            self.reinforce(iterations=20, units=ii)

            data = self.buffer.get_sample(self.buffer.size, remove=True)
            self.policy_net.retrain_model(data)
            self.value_net.retrain_model(data)
            self.buffer.increase_max_size(0.1)

        for ii in range(10):
            print("Training with all units. Currently on iteration ", str(ii+1))
            bm = Benchmark()
            bm.benchmark_caserta()
            self.reinforce(iterations=10, units=self.height*self.width)
            self.train_and_update_models()

        # find best parameters
        # run experiment on test instances

    def produce_testing_data(self, filename, examples=10000, perm=False):
        data_list = []
        start = time.time()
        for e in range(examples):
            if e % 500 == 0 and e > 0:
                h = not os.path.isfile(filename)
                final_df = pd.concat(data_list)
                end = time.time()
                with open(filename, 'a+') as f:
                    final_df.to_csv(f, header=h, index=False)
                print(end - start)

                start = time.time()
                data_list = []

            data, length = self.create_training_example(permutations=perm, units=self.height*self.width)
            data_list.append(data)

        # in case number is not divisible by 500
        final_df = pd.concat(data_list)
        with open(filename, 'a') as f:
            final_df.to_csv(f, header=False, index=False)


if __name__ == "__main__":
    test = Optimizer()
    #test.test_combinded_model()
    #test.reinforce(10)
    #test.test_value_network()
    #test.train_on_new_instances(1)
    #test.training_on_csv()
    #test.evaluate_parameters()
    #test.test_wrapper()
    #test.find_best_parameters()
    test.full_experiment()
    #test.test_stupid_wrapper()
    #test.test_keras()
    #test.produce_testing_data(filename="test_timo.csv", examples=1000, perm=False)
    #test.train_on_csv("up_to_8.csv")
    #test.test_deviations()

