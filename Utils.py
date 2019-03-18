import json
import pickle


def load_configs(filename):
    with open(filename) as json_data:
        return json.load(json_data)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


