import json
import pickle
import numpy as np
import glob


def load_configs(filename):
    with open(filename) as json_data:
        return json.load(json_data)


def load_obj(name):
    try:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        d = {}
        for ii in range(-99, 0):
            d[ii] = 1
        return d


def save_obj(file, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


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
    starting_string = "data" + str(height) + "-" + str(width) + "-"
    file_names = [starting_string + str(x) + ".dat" for x in range(1,41)]

    matrices = []
    for f in file_names:
        f = "OfficalBenchmarkInstances/CRPTestcases_Caserta/" + f
        matrices.append(load_benchmark_instances(f, height))

    return matrices


if __name__ == "__main__":
    load_caserta(3,3)
    a = load_benchmark_instances("OfficalBenchmarkInstances/CRPTestcases_Caserta/data3-3-1.dat", 3)
    print(a)