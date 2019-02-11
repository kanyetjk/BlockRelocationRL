import numpy as np
import tensorflow as tf

# TODO Load config from json


class ApproximationModel(object):
    """The parent class to the value and policy network.

    The Deep Learning architecture is using the tensorflow estimator API where a model_fn has to be implemented.
    The parent class implements the beginning of the network up to the output layer, which is implemented in the
    child classes. The model architecture can be changed in the configs.

    Attributes:
        TODO attributes
    """

    def __init__(self, configs):
        self.height = configs["height"] + 2
        self.width = configs["width"]
        self.model = None  # TODO?
        self.steps_list = []
        self.max_value = configs["height"] * configs["width"]

        self.num_epochs = 1
        self.batch_size = configs["batch_size"]
        self.learning_rate = configs["learning_rate"]
        self.shared_weights = configs["shared_weights"]
        self.connected_layers = configs["fully_connected"]
        self.name = "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "_SH_" \
                    + str(self.shared_weights[0]) + "_FC_" + str(self.connected_layers[0])

        self.current_model_size = self.width * self.height

    def build_model_beginning(self, features):
        current_layer = features["x"]
        self.current_model_size = self.width * self.height
        for i, size in enumerate(self.shared_weights):
            current_layer = self.shared_layer(num_hidden=size, input_tensor=current_layer, layer_count=i)

        for i, size in enumerate(self.connected_layers):
            current_layer = self.hidden_layer(num_hidden=size, input_tensor=current_layer, relu=True, layer_count=i)

        return current_layer

    def shared_layer(self, num_hidden, input_tensor, layer_count):
        with tf.name_scope("Shared_layer" + str(layer_count)):
            neurons = int(self.current_model_size / self.width)
            init = tf.Variable(tf.random_normal([neurons, num_hidden], dtype=tf.float64))
            weights = tf.get_variable(name="weights" + str(layer_count), initializer=init)

            init = tf.Variable(tf.zeros([num_hidden], dtype=tf.float64))
            biases = tf.get_variable(name="bias" + str(layer_count), initializer=init)

            tensor_list = []
            for x in range(self.width):
                col = tf.slice(input_tensor, [0, neurons * x], [-1, neurons])
                layer = tf.nn.relu(tf.matmul(col, weights) + biases)
                tensor_list.append(layer)

            combination_layer = tf.concat(tensor_list, name="Combination" + str(layer_count), axis=1)
            self.current_model_size = num_hidden * self.width
            return combination_layer

    def hidden_layer(self, num_hidden, input_tensor, relu=True, layer_count=0):
        with tf.name_scope("Hidden_Layer" + str(layer_count)):
            bias = tf.Variable(tf.random_normal([num_hidden], dtype=tf.float64))
            weights = tf.Variable(tf.random_normal([self.current_model_size, num_hidden], dtype=tf.float64))
            if relu:
                layer = tf.nn.relu(tf.add(tf.matmul(input_tensor, weights), bias))
            else:
                layer = tf.add(tf.matmul(input_tensor, weights), bias)
        self.current_model_size = num_hidden
        return layer

    def prepare_state_data(self, data):
        prepared_data = data.StateRepresentation.values
        prepared_data = np.array([np.array(x) / self.max_value for x in prepared_data])
        return prepared_data

    def input_fn_train(self, x, y):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x}, y=y,
            batch_size=self.batch_size, num_epochs=self.num_epochs, shuffle=True)
        return input_fn

    def input_fn_predict(self, x):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x},
            batch_size=self.batch_size, num_epochs=1, shuffle=False)
        return input_fn

    def input_fn_evaluate(self, x, y):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': x}, y=y,
            batch_size=self.batch_size, num_epochs=self.num_epochs, shuffle=False)
        return input_fn

    def train(self, x, y):
        self.model.train(self.input_fn_train(x, y))

    def predict_df(self, data):
        X = data.StateRepresentation
        X = [x.transpose().flatten() for x in X]
        X = np.array([x / self.max_value for x in X])
        return self.predict(X)

    def predict(self, x):
        return self.model.predict(self.input_fn_predict(x))

    def evaluate(self, x, y):
        return self.model.evaluate(self.input_fn_train(x, y, batch_size=128))

    def evaluate_df(self, df):
        X = self.prepare_state_data(df)
        y = df.Value
        y = np.array([np.array([val], dtype=float) for val in y])
        self.evaluate(X, y)

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
    def __init__(self, configs):
        super().__init__(configs)
        path = "TensorBoardFiles/PN" + self.name + "/"
        self.model = tf.estimator.Estimator(self.model_fn, path)

    def train_df(self, df):
        X = self.prepare_state_data(df)

        y = df.MovesEncoded
        y = np.array([np.array(val, dtype=float) for val in y])
        self.train(X, y)

    def model_fn(self, features, labels, mode):
        last_layer = self.build_model_beginning(features)

        num_output = self.width * (self.width-1)
        output_layer = self.hidden_layer(num_hidden=num_output, input_tensor=last_layer, relu=False)

        percent_output = tf.nn.softmax(output_layer)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=percent_output)

        loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output_layer)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        #TODO
        #acc_op = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=percent_output)
        classes = tf.math.argmax(labels,axis=1)
        binary_prediction = tf.math.argmax(percent_output)
        #accuracy = tf.metrics.accuracy(labels=labels, predictions=binary_prediction)
        mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(labels=classes, predictions=percent_output,
                                                                     num_classes=12)
        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=percent_output,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': mean_per_class_accuracy})

        return estimator_specs


class ValueNetwork(ApproximationModel):
    def __init__(self, configs):
        super().__init__(configs)
        path = "TensorBoardFiles/VN" + self.name + "/"
        self.model = tf.estimator.Estimator(self.model_fn, path)

    def model_fn(self, features, labels, mode):
        last_layer = self.build_model_beginning(features)

        predicted_value = self.hidden_layer(num_hidden=1, input_tensor=last_layer, relu=False)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predicted_value)

        loss_op = tf.losses.mean_squared_error(labels=labels,
                                               predictions=predicted_value)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

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
        X = self.prepare_state_data(df)

        y = df.Value
        y = np.array([np.array([val], dtype=float) for val in y])
        self.train(X, y)
