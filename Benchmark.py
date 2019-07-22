from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from Utils import load_configs, load_caserta
from KerasModel import ValueNetworkKeras, PolicyNetworkKeras

import numpy as np
import time
import logging
import logging.handlers


class Benchmark(object):
    def __init__(self):
        configs = load_configs("Configs.json")
        value_configs = load_configs("Configs_ValueNN.json")
        policy_configs = load_configs("Configs_PolicyNN.json")

        self.width = configs["width"]
        self.height = configs["height"]
        self.env = BlockRelocation(self.height, self.width)

        self.value_net = ValueNetworkKeras(value_configs)
        self.policy_net = PolicyNetworkKeras(policy_configs)

        self.tree_searcher = TreeSearch(self.value_net, BlockRelocation(self.height, self.width), self.policy_net)

    def benchmark(self, filename, function, params, algorithm_type):
        matrices = self.load_benchmark_instances(filename)
        times = []
        steps = []
        for i, m in enumerate(matrices):
            if i > 1 and i % 10 == 0:
                print("Done with ", str(i))
            start = time.time()
            num_steps = len(function(m, **params))
            stop = time.time()
            times.append(stop-start)
            steps.append(num_steps)
        print(times, steps)

        # logging the results
        self.log_results(times, steps, algorithm_type, params)

    @staticmethod
    def log_results(times, steps, func_name, params):
        LOG_FILENAME = "BenchmarkLogs/" + func_name + str(round(time.time())) + ".log"

        # Set up a specific logger with our desired output level
        my_logger = logging.getLogger('MyLogger')
        my_logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILENAME)
        my_logger.addHandler(handler)

        # Overview of the benchmark run
        my_logger.info("=== Results of Benchmark ===")
        my_logger.info("Date: {}".format(str(time.asctime())))
        my_logger.info("Function Type: {}".format(func_name))
        my_logger.info("Parameter: {}".format(params))
        my_logger.info("Total Time: {}".format(round(np.sum(times), 2)))
        my_logger.info("Total Steps: {}".format(np.sum(steps)))
        my_logger.info("Max Time: {}".format(round(max(times), 2)))
        my_logger.info("Average Steps: {}".format(round(np.mean(steps), 2)))

        my_logger.info("Average Time: {}".format(round(np.mean(times), 2)))

        # Individual steps and times
        my_logger.info("\n=== Individual Solutions ===")
        for i, t, num_steps in zip(range(len(times)), times, steps):
            my_logger.info("Instance: {},   Steps: {},  time: {}".format(i, num_steps, t))

    @staticmethod
    def load_benchmark_instances(filename):
        return np.load(filename)

    def benchmark_caserta(self):
        matrices = load_caserta(self.width, self.height)
        function = self.tree_searcher.find_path_dfs
        params = {"stop_param": 1, "k": 10}
        algorithm_type = "dfs"

        times = []
        steps = []
        for i, m in enumerate(matrices):
            if i > 1 and i % 10 == 0:
                print("Done with ", str(i))
            start = time.time()
            num_steps = len(function(m, **params))
            stop = time.time()
            print(num_steps, stop-start)
            times.append(stop-start)
            steps.append(num_steps)
        print(times, steps)

        # logging the results
        self.log_results(times, steps, algorithm_type, params)

    def test_performance(self, units):
        matrices = [self.env.create_instance_random(units) for _ in range(5)]

        start = time.time()
        num_steps = 0
        for m in matrices:
            print("abc")
            num_steps += len(self.tree_searcher.find_path_dfs(m, stop_param=2, k=15, cutoff_param=0.002))
        stop = time.time()
        print(stop-start)
        print(num_steps)

        start = time.time()
        num_steps = 0
        for m in matrices:
            print("abc")
            num_steps += len(self.tree_searcher.find_path_dfs(m, stop_param=2, k=15, cutoff_increase=1.4,
                                                              cutoff_param=0.002))
        stop = time.time()
        print(stop - start)
        print(num_steps)

    def test_params(self):
        params = [1,2,3,4,5]
        k = 12

        matrices = load_caserta(self.width, self.height)
        function = self.tree_searcher.find_path_dfs

        for p in params:
            params = {"stop_param": p, "k": k}
            times = []
            steps = []

            for i, m in enumerate(matrices):
                if i > 1 and i % 10 == 0:
                    print("Done with ", str(i))
                start = time.time()
                num_steps = len(function(m, **params))
                stop = time.time()
                times.append(stop - start)
                steps.append(num_steps)
            print("Param: " + str(p), "Steps: " + str(sum(steps)) + "Time: " + str(sum(times)))


if __name__ == "__main__":
    bm = Benchmark()
    #bm.benchmark_caserta()
    bm.test_performance(20)

