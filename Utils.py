import json


def load_configs():
    with open("Configs.json") as json_data:
        return json.load(json_data)
