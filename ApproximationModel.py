import numpy as np
import tensorflow as tf


# TODO Batch size bigger than 1
# TODO Load config from json


class ApproximationModel(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.model = None  # TODO?
        self.steps_list = []
        pass
        # TODO load configs

    def build_model_beginning(self, features):
        input_tensor = features["x"]
        shared_layer = self.sharded_layer(num_columns=self.width, column_height=self.height, num_hidden=3,
                                          input_tensor=input_tensor)
        first_layer = self.hidden_layer(n_input=self.width * 3, n_hidden=100, name_scope="first_connected_layer",
                                        prev_layer=shared_layer)
        second_layer = self.hidden_layer(n_input=100, n_hidden=100, name_scope="second_layer",
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
                col = tf.slice(input_tensor, [0, column_height * x], [-1, column_height])
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
    def input_fn_train(x, y, batch_size=1, num_epochs=5):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x}, y=y,
            batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
        return input_fn

    @staticmethod
    def input_fn_test(x):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x},
            batch_size=128, num_epochs=1, shuffle=False)
        return input_fn

    def train(self, x, y):
        # TODO HOOKS?
        self.model.train(self.input_fn_train(x, y, batch_size=128, num_epochs=2))

    def predict_df(self, x):
        X = x.StateRepresentation
        X = np.array([x.transpose().flatten() / 100 for x in X])
        return self.predict(X)

    def predict(self, x):
        # this will be a generator??
        return self.model.predict(self.input_fn_test(x))

    def evaluate(self, x, y):
        return self.model.evaluate(self.input_fn_train(x, y, batch_size=1, num_epochs=2))

    def write_steps_summary(self, steps):
        # TODO CHANGE BACK TO 10
        self.steps_list.append(steps)
        if len(self.steps_list) >= 1:
            print(steps)
            average_steps = np.mean(self.steps_list)
            self.steps_list = []
            writer = tf.summary.FileWriter('TBGraphs/')
            summary = tf.Summary()
            summary.value.add(tag='AverageMoves', simple_value=average_steps)
            writer.add_summary(summary)

            writer.flush()
            writer.close()


class PolicyNetwork(ApproximationModel):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.model = tf.estimator.Estimator(self.model_fn, "GraphPN/")

    def train_df(self, df):
        X = df.StateRepresentation.values
        X = np.array([np.array(x) for x in X])

        y = df.MovesEncoded
        y = np.array([np.array(val, dtype=float) for val in y])
        self.train(X, y)

    def input_fn_train(self, x, y, batch_size=128, num_epochs=1):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x}, y=y,
            batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
        return input_fn

    def train(self, x, y):
        self.model.train(self.input_fn_train(x, y))

    def model_fn(self, features, labels, mode):
        # TODO NOT CLEAR WITH THE OUTPUT
        last_layer = self.build_model_beginning(features)

        num_output = self.width * (self.width-1)
        output_layer = self.hidden_layer(n_input=100, n_hidden=num_output, prev_layer=last_layer, name_scope="output",
                                         relu=False)

        percent_output = tf.nn.softmax(output_layer)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=percent_output)

        loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=percent_output)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        acc_op = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=percent_output)

        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=percent_output,
            loss=loss_op,
            train_op=train_op)

        return estimator_specs


class ValueNetwork(ApproximationModel):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.model = tf.estimator.Estimator(self.model_fn, "TBGraphs/")

    def model_fn(self, features, labels, mode):
        last_layer = self.build_model_beginning(features)

        predicted_value = self.hidden_layer(n_input=100, n_hidden=1, prev_layer=last_layer, name_scope="output",
                                            relu=False)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predicted_value)

        loss_op = tf.losses.mean_squared_error(labels=labels,
                                               predictions=predicted_value)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        acc_op = tf.metrics.mean_absolute_error(labels=labels, predictions=predicted_value)

        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predicted_value,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'mean_abs_error': acc_op})

        return estimator_specs

    def train_df(self, df):
        X = df.StateRepresentation.values
        X = np.array([np.array(x) for x in X])
        #X = np.array([x.transpose().flatten() / 100 for x in X])

        y = df.Value
        y = np.array([np.array([val], dtype=float) for val in y])
        self.train(X, y)


if __name__ == "__main__":
    test = PolicyNetwork(4,6)
