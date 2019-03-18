from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from Utils import load_configs
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

        logging.basicConfig(format='%(asctime)s %(message)s', filename=configs["log_name"], level=logging.WARNING,
                            datefmt="%Y-%m-%d %H:%M:%S")
        logging.info("Starting Optimizer.")

        self.width = configs["width"]
        self.height = configs["height"]
        self.env = BlockRelocation(self.height, self.width)

        self.value_net = ValueNetworkKeras(value_configs)
        self.policy_net = PolicyNetworkKeras(policy_configs)

        self.tree_searcher = TreeSearch(self.value_net, BlockRelocation(self.height, self.width), self.policy_net)


    def create_benchmark_instances(self, n):
        matrices = [self.env.create_instance(self.height, self.width) for _ in range(n)]
        np.save("BenchmarkInstances/4x4", matrices)

    def benchmark(self, filename, function, params, algorithm_type):
        matrices = self.load_benchmark_instances(filename)
        times = []
        steps = []
        for i, m in enumerate(matrices):
            if i > 1 and i % 10 == 0:
                print(f"Done with {i}")
            start = time.time()
            num_steps = len(function(m, **params))
            stop = time.time()
            times.append(stop-start)
            steps.append(num_steps)
        print(times, steps)

        # logging the results
        self.log_results(times, steps, algorithm_type, params)

    def log_results(self, times, steps, func_name, params):
        LOG_FILENAME = "BenchmarkLogs/" +func_name + str(round(time.time())) + ".log"

        # Set up a specific logger with our desired output level
        my_logger = logging.getLogger('MyLogger')
        my_logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILENAME)
        my_logger.addHandler(handler)

        my_logger.info("=== Results of Benchmark ===")
        my_logger.info(f"Function Type: {func_name}")
        my_logger.info(f"Parameter: {params}")
        my_logger.info(f"Total Time: {round(np.sum(times), 2)} s")
        my_logger.info(f"Total Steps: {np.sum(steps)} steps")
        my_logger.info(f"Max Time: {round(max(times), 2)}")
        my_logger.info(f"Average Steps: {round(np.mean(steps), 2)}")
        my_logger.info(f"Average Time: {round(np.mean(times), 2)} s")

        my_logger.info("\n=== Individual Solutions ===")
        for i, t, num_steps in zip(range(len(times)), times, steps):
            my_logger.info(f"Instance: {i},   Steps: {num_steps},  time: {round(t,2)}")

    def load_benchmark_instances(self, filename):
        return np.load(filename)

    def benchmark_dfs(self):
        func = self.tree_searcher.find_path_dfs
        params = {"stop_param": 1.5, "k": 7}
        self.benchmark("BenchmarkInstances/4x4.npy", func, params, "DFS")

    def benchmark_bfs(self):
        func = self.tree_searcher.find_path_2
        params = {"search_depth": 5,
                  "epsilon": 0.1,
                  "threshold": 0.01,
                  "drop_percent": 0.25,
                  "factor": 0.01}
        self.benchmark("BenchmarkInstances/4x4.npy", func, params, "BFS")

bm = Benchmark()
bm.benchmark_dfs()
