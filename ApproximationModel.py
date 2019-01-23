import tensorflow as tf
import numpy as np


# TODO Load config from json

class ApproximationModel:
    def __init__(self):
        self.model = None  # TODO?
        pass
        # TODO load configs

    def build_model_beginning(self, features):
        input_tensor = features["x"]
        shared_layer = self.sharded_layer(num_columns=2, column_height=5, num_hidden=3, input_tensor=input_tensor)
        first_layer = self.hidden_layer(n_input=6, n_hidden=20, name_scope="first_connected_layer",
                                        prev_layer=shared_layer)
        second_layer = self.hidden_layer(n_input=20, n_hidden=20, name_scope="second_layer",
                                         prev_layer=first_layer)
        return second_layer

    @staticmethod
    def reshape_input(input_matrix):
        return input_matrix.transpose().flatten()

    def sharded_layer(self, num_columns, column_height, num_hidden, input_tensor):
        with tf.name_scope("Shared_layer"):
            init = tf.Variable(tf.random_normal([column_height, num_hidden], dtype=tf.float64))
            weights = tf.get_variable('Shared_weights', initializer=init)

            init = tf.Variable(tf.zeros([num_hidden], dtype=tf.float64))
            biases = tf.get_variable('Shared_biases', initializer=init)

            tensor_list = []
            for x in range(num_columns):
                col = tf.slice(input_tensor, [0, column_height * x], [1, column_height])
                name = "column_" + str(x)
                layer = tf.nn.relu(tf.matmul(col, weights) + biases)
                next_layer = self.hidden_layer(n_input=num_hidden, n_hidden=3, prev_layer=layer, name_scope=name)
                tensor_list.append(next_layer)

            combination_layer = tf.concat(tensor_list, name="Combination", axis=1)
            return combination_layer

    @staticmethod
    def hidden_layer(n_input, n_hidden, prev_layer, name_scope, relu=True):
        with tf.name_scope(name_scope):
            bias = tf.Variable(tf.random_normal([n_hidden], dtype=tf.float64))
            weights = tf.Variable(tf.random_normal([n_input, n_hidden], dtype=tf.float64))
            if relu:
                layer = tf.nn.relu(tf.add(tf.matmul(prev_layer, weights), bias))
            else:
                layer = tf.add(tf.matmul(prev_layer, weights), bias)
        return layer

    @staticmethod
    # TODO set epochs global?
    def input_fn_train(x, y, batch_size=128, num_epochs=1):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x}, y=y,
            batch_size=batch_size, num_epochs=100000, shuffle=True)
        return input_fn

    @staticmethod
    def input_fn_test(x):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x},
            batch_size=128, num_epochs=1, shuffle=True)
        return input_fn

    def train(self, x, y):
        # TODO HOOKS?
        self.model.train(self.input_fn_train(x, y, batch_size=128, num_epochs=2))

    def predict(self, x):
        # this will be a generator??
        return self.model.predict(self.input_fn_test(x))


class PolicyNetwork(ApproximationModel):
    def __init__(self):
        super().__init__()
        self.model = tf.estimator.Estimator(self.model_fn, "TBGraphs/")
        self.model.train(self.input_fn_train(x, y))

    def model_fn(self, features, labels, mode):
        pass


class ValueNetwork(ApproximationModel):
    def __init__(self):
        super().__init__()
        x = np.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], np.ones(10)], dtype=float)
        y = np.array([[1], [1]], dtype=float)
        self.model = tf.estimator.Estimator(self.model_fn, "TBGraphs/")
        # self.model.train(self.input_fn_train(x, y))

        # test = self.model.predict(self.input_fn_test(x))
        # eval = self.model.evaluate(self.input_fn_train(x, y))
        # print(eval)
        # print(list(test))

    def model_fn(self, features, labels, mode):
        last_layer = self.build_model_beginning(features)

        logits = self.hidden_layer(n_input=20, n_hidden=1, prev_layer=last_layer, name_scope="output", relu=False)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=logits)

        loss_op = tf.losses.mean_squared_error(labels=labels,
                                               predictions=logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=1)

        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        acc_op = tf.metrics.mean_absolute_error(labels=labels, predictions=logits)

        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=logits,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'mean_abs_error': acc_op})

        return estim_specs


test = ValueNetwork()
