from KerasModel import ValueNetworkKeras, PolicyNetworkKeras
import pandas as pd
import numpy as np
import logging

formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# first file logger
eval_logger = setup_logger('evluation', 'evaluation.log')
tf_logger = setup_logger('tensorflow', 'tensorflow.log')


possible_parameters = {
    "learning_rate": [0.01, 0.005, 0.001],
    "batch_size": [64, 128],
    "shared_weights": [[32, 32], [32, 16], [16, 16], [16, 8]],
    "fully_connected": [[256, 256, 128], [256, 128, 128], [128, 128, 64], [64, 64, 32], [32, 32, 16]]
}

#load data
df = pd.read_csv("up_to_8.csv")
df["StateRepresentation"] = df["StateRepresentation"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
df["MovesEncoded"] = df["MovesEncoded"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
df["hashed"] = df["StateRepresentation"].apply(lambda s: s.tostring())
df = df.sort_values('Value', ascending=False).drop_duplicates('hashed').sort_index()
# train test split

test_data = df.sample(frac=0.2)
df = df.drop(test_data.index)
train_data = df
print("Loading Data Complete")
print(train_data.shape)

for x in range(1):
    configs = {
        "width": 4,
        "height": 4,
        "experiment": True,
        "learning_rate": 0.001,
        "batch_size": 128,
        "shared_weights": [16, 16],
        "fully_connected": [64, 64, 32],
        "num_epochs": True
    }

    current_model = ValueNetworkKeras(configs)
    #current_model.train_experiment(train_data)
    loss, mae = current_model.eval(test_data)
    eval_logger.info(configs)
    eval_logger.info(f"MAE: {mae}")
