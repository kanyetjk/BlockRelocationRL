import json
import pickle
import numpy as np
import glob


def load_configs(filename):
    with open(filename) as json_data:
        return json.load(json_data)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_benchmark_instances(filename, height):
    m = []
    with open(filename) as f:
        next(f)
        for line in f:
            m.append([int(x) for x in line.split()[1:]])

    for col in m:
        if len(col) < height:
            for _ in range(height-len(col)):
                col.append(0)

    m = np.flip(np.array(m).transpose(), axis=0)
    padding = np.zeros([2,height])
    m = np.concatenate((padding, m), axis=0)
    return m


def load_caserta(width, height):
    import os
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'OfficalBenchmarkInstances/CRPTestcases_Caserta')
    starting_string = "data" + str(height) + "-" + str(width)
    file_names = [fn for fn in os.listdir(filename) if fn.startswith(starting_string)]

    matrices = []
    for f in file_names:
        f = "OfficalBenchmarkInstances/CRPTestcases_Caserta/" + f
        matrices.append(load_benchmark_instances(f, height))
    return matrices


if __name__ == "__main__":
    load_caserta(4, 4)
    load_benchmark_instances("OfficalBenchmarkInstances/CRPTestcases_Caserta/data4-4-9.dat", 4)