from BlockRelocation import BlockRelocation
from TreeSearch import TreeSearch
from Utils import load_configs, load_caserta
from KerasModel import ValueNetworkKeras, PolicyNetworkKeras
import ray
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.track import log
import ray.tune

import numpy as np
import time
import logging
import logging.handlers

class Opt(object):
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
        self.matrices = []


A = Opt()

matrices = [A.env.create_instance_random(10) for _ in range(10)]


def easy_objective(config):
    steps = 0
    """
    for j, m in enumerate(matrices):
        f = A.tree_searcher.find_path_dfs
        num_steps = len(f(m, **config))
        print(num_steps)
        if steps == 0:
            num_steps = 100
        steps += num_steps"""
    for i in config["iterations"]:
        steps = config["cutoff_param"]*1000 + config["cutoff_param"]*8
        ray.tune.track.log(steps=steps)

def main():
    #matrices = [self.env.create_instance_random(10) for _ in range(10)]
    ray.init()
    space = {#"k": ray.tune.sample_from([1,2,3,4]),
             "stop_param": (0, 10),
             "cutoff_param": (0.0005, 0.1)}

    config = {"config": {"iterations": 20}}

    algo = BayesOptSearch(space,
                          max_concurrent=10,
                          metric="steps",
                          mode="min",
                          utility_kwargs={
                              "kind": "ucb",
                              "kappa": 2.5,
                              "xi": 0
                          })

    scheduler = AsyncHyperBandScheduler(metric="steps", mode="min")
    analysis = run(easy_objective,
        name="test1",
        search_alg=algo,
        scheduler=scheduler,
        **config)




if __name__ == "__main__":
    main()
