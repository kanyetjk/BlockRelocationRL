from keras.models import Model, load_model
from keras.layers import Dense, Input, Concatenate, Lambda, regularizers, LeakyReLU, Softmax
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


class KerasModel(object):
    def __init__(self, configs):
        self.shared_weights = configs["shared_weights"]
        self.fully_connected = configs["fully_connected"]
        self.learning_rate = configs["learning_rate"]
        self.batch_size = configs["batch_size"]
        self.epochs = configs["num_epochs"]

        self.width = configs["width"]
        self.height = configs["height"] + 2

        self.model = self.build_model()
        self.dir = self.generate_tensorboard_name()
        if "experiment" in configs:
            self.dir += "_EXP"

        self.model = self.build_model()
        try:
            self.model.load_weights(filepath=self.dir + "/model.hdf5")
            print("Loading existing model.")
            print(self.dir + "/model.hdf5")
        except OSError:
            print("No existing model found.")

        self.tensorboard = TensorBoard(log_dir=self.dir, histogram_freq=0,
                                       write_graph=True, write_images=False)

        self.saver = ModelCheckpoint(filepath=self.dir + "/model.hdf5", verbose=False)

        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
                                            baseline=None, restore_best_weights=True)

    def reset_model(self):
        self.model = self.build_model()

    def retrain_model(self, data):
        self.reset_model()
        self.train_experiment(data)

    def train_experiment(self, data):
        pass

    def build_model(self):
        pass

    def generate_tensorboard_name(self):
        pass

    def predict_df(self, data):
        # preparing data
        max_val = (self.height - 2) * self.width
        x_data = data.StateRepresentation.values
        x_data = np.array([x.transpose().flatten() / max_val for x in x_data])

        return self.model.predict(x_data, batch_size=256)

    def predict_single(self, matrix):
        max_val = (self.height - 2) * self.width
        x_data = np.array([matrix.transpose().flatten() / max_val])
        return self.model.predict(x_data)[0]

    @staticmethod
    def load_file(filename):
        data = pd.read_csv(filename)
        print(data.shape)
        data["StateRepresentation"] = data["StateRepresentation"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
        data["MovesEncoded"] = data["MovesEncoded"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))

        data["hashed"] = data["StateRepresentation"].apply(lambda s: s.tostring())
        data = data.drop_duplicates(subset="hashed")
        data = data.drop(columns=["hashed"])
        data = data.reset_index(drop=True)
        return data


class ValueNetworkKeras(KerasModel):
    def __init__(self, configs):
        super().__init__(configs)

    def generate_tensorboard_name(self):
        name = ["TensorBoardFiles/KVN"]
        name += [str(x) for x in [self.width, self.height]]
        name += ["SW"]
        name += [str(x) for x in self.shared_weights]
        name += ["FC"]
        name += [str(x) for x in self.fully_connected]
        name += [str(x) for x in [self.learning_rate, self.batch_size]]
        name = "_".join(name)
        return name

    def build_model(self):
        inputArray = Input(shape=(self.width * self.height,))

        layer = inputArray

        shared_dense = Dense(self.shared_weights[0], activation='relu')
        layerArray = []
        h = self.height
        for t in range(self.width):
            out = Lambda(lambda x: x[:, t*h:(t+1)*h])(layer)
            layerArray.append(shared_dense(out))

        layer = layerArray

        for neurons in self.shared_weights[1:]:
            shared_dense = Dense(neurons, activation='relu')
            layerArray = []
            for t in range(self.width):
                layerArray.append(shared_dense(layer[t]))
            layer = layerArray

        merged_vector = Concatenate(axis=-1)(layer)

        layer = merged_vector
        for neurons in self.fully_connected:
            layer = Dense(neurons, activation='relu')(layer)
        output_layer = Dense(1, activation='linear')(layer)
        model = Model(input=inputArray, output=output_layer)

        adam = Adam(lr=self.learning_rate)
        model.compile(optimizer=adam, loss='mse', metrics=['mae'])
        return model

    def train_df(self, data, epochs=None, validation=True):
        if not epochs:
            epochs = self.epochs

        validation_split = 0.15 if validation else 0

        # preparing data
        max_val = (self.height - 2) * self.width
        x_data = data.StateRepresentation.values
        x_data = np.array([x / max_val for x in x_data])

        y = data.Value.values
        y = np.array([np.array([val], dtype=float) for val in y])

        # fit the model
        self.model.fit(x_data, y,
                       batch_size=self.batch_size,
                       epochs=epochs,
                       shuffle=True,
                       callbacks=[self.tensorboard, self.saver],
                       validation_split=validation_split,
                       verbose=0)

    def train_experiment(self, data):
        max_val = (self.height - 2) * self.width
        x_data = data.StateRepresentation.values
        x_data = np.array([x / max_val for x in x_data])

        y = data.Value.values
        y = np.array([np.array([val], dtype=float) for val in y])

        # fit the model
        self.model.fit(x_data, y,
                       batch_size=self.batch_size,
                       epochs=50,
                       shuffle=True,
                       callbacks=[self.tensorboard, self.saver, self.early_stopping],
                       validation_split=0.1,
                       verbose=1)

    def eval(self, data):
        max_val = (self.height - 2) * self.width
        x_data = data.StateRepresentation.values
        x_data = np.array([x / max_val for x in x_data])

        y = data.Value.values
        y = np.array([np.array([val], dtype=float) for val in y])

        return self.model.evaluate(x_data, y,
                                   batch_size=256)


class PolicyNetworkKeras(KerasModel):
    def __init__(self, configs):
        super().__init__(configs)

    def generate_tensorboard_name(self):
        name = ["TensorBoardFiles/KPN"]
        name += [str(x) for x in [self.width, self.height]]
        name += ["SW"]
        name += [str(x) for x in self.shared_weights]
        name += ["FC"]
        name += [str(x) for x in self.fully_connected]
        name += [str(x) for x in [self.learning_rate, self.batch_size]]
        name = "_".join(name)
        return name

    def build_model(self):
        # input layer
        input_array = Input(shape=(self.width * self.height,))
        layer = input_array

        # first shared layer
        shared_dense = Dense(self.shared_weights[0], activation='relu')
        layer_array = []
        h = self.height
        for t in range(self.width):
            out = Lambda(lambda x: x[:, t*h:(t+1)*h])(layer)
            layer_array.append(shared_dense(out))

        layer = layer_array

        # the other shared layers
        for neurons in self.shared_weights[1:]:
            shared_dense = Dense(neurons, activation='relu')
            layer_array = []
            for t in range(self.width):
                layer_array.append(shared_dense(layer[t]))
            layer = layer_array

        merged_vector = Concatenate(axis=-1)(layer)

        # fully connected layers
        layer = merged_vector
        for neurons in self.fully_connected:
            layer = Dense(neurons, activation='relu')(layer)

        # output layer
        num_output = self.width * (self.width-1)
        output_layer = Dense(num_output, activation='linear')(layer)
        output_layer = LeakyReLU(alpha=0.3)(output_layer)
        output_layer = Softmax()(output_layer)
        model = Model(input=input_array, output=output_layer)

        # optimizer, loss, metrics
        adam = Adam(lr=self.learning_rate)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        return model

    def train_df(self, data, epochs=None, validation=True):
        if not epochs:
            epochs = self.epochs

        validation_split = 0.15 if validation else 0

        # preparing data
        max_val = (self.height - 2) * self.width
        x_data = data.StateRepresentation.values
        x_data = np.array([x / max_val for x in x_data])

        y = data.MovesEncoded
        y = np.array([np.array(val, dtype=float) for val in y])

        # fit the model
        self.model.fit(x_data, y,
                       batch_size=self.batch_size,
                       epochs=epochs,
                       shuffle=True,
                       callbacks=[self.tensorboard, self.saver],
                       validation_split=validation_split,
                       verbose=0)

    def train_experiment(self, data, epochs=None, validation=True):
        # preparing data
        max_val = (self.height - 2) * self.width
        x_data = data.StateRepresentation.values
        x_data = np.array([x / max_val for x in x_data])

        y = data.MovesEncoded
        y = np.array([np.array(val, dtype=float) for val in y])

        # fit the model
        self.model.fit(x_data, y,
                       batch_size=self.batch_size,
                       epochs=50,
                       shuffle=True,
                       callbacks=[self.tensorboard, self.saver, self.early_stopping],
                       validation_split=0.1,
                       verbose=1)

    def eval(self, data):
        max_val = (self.height - 2) * self.width
        x_data = data.StateRepresentation.values
        x_data = np.array([x / max_val for x in x_data])

        y = data.MovesEncoded
        y = np.array([np.array(val, dtype=float) for val in y])

        return self.model.evaluate(x_data, y,
                                   batch_size=256)


class ModelExperiment:
    def __init__(self):
        self.filename = "abc"
        configs = {}
        self.policy_network = PolicyNetworkKeras(configs)
        self.value_network = ValueNetworkKeras(configs)

    def run_experiment(self):
        data = KerasModel.load_file(self.filename)
        train_data, test_data = train_test_split(data, shuffle=True, test_size=0.15)

        # try policy network
        self.policy_network.train_experiment(train_data)
        # TODO LOG
        self.policy_network.eval(test_data)

        # try value network
        self.value_network.train_experiment(train_data)
        # TODO LOG
        self.value_network.eval(test_data)


if __name__ == "__main__":
    test = ModelExperiment()
