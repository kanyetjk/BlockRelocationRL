import json


def load_configs(filename):
    with open(filename) as json_data:
        return json.load(json_data)
