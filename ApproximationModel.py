import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

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

        self.num_epochs = configs["num_epochs"]
        self.batch_size = configs["batch_size"]
        self.learning_rate = configs["learning_rate"]
        self.shared_weights = configs["shared_weights"]
        self.connected_layers = configs["fully_connected"]
        self.name = self.generate_tensorboard_name()

        self.current_model_size = self.width * self.height

    def generate_tensorboard_name(self):
        name = []
        name += [str(x) for x in [self.width, self.height]]
        name += ["SW"]
        name += [str(x) for x in self.shared_weights]
        name += ["FC"]
        name += [str(x) for x in self.connected_layers]
        name += [str(x) for x in [self.learning_rate, self.batch_size]]
        name = "_".join(name)
        return name

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
            batch_size=self.batch_size, num_epochs=1, shuffle=False)
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
        #hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline")]
        #return self.model.evaluate(self.input_fn_evaluate(x, y), hooks=hooks)
        return self.model.evaluate(self.input_fn_evaluate(x, y))

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
        output_layer = tf.nn.leaky_relu(output_layer)

        percent_output = tf.nn.softmax(output_layer)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=percent_output)

        loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=output_layer)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        policy_acc = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=labels)
        policy_acc = tf.metrics.mean(policy_acc)

        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=percent_output,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'cross_entropy': policy_acc})

        return estimator_specs

    def evaluate_df(self, df):
        X = self.prepare_state_data(df)
        y = df.MovesEncoded
        y = np.array([np.array(val, dtype=float) for val in y])
        self.evaluate(X, y)


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


class CombinedModel(ApproximationModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.value_head = configs["value_head"]
        self.policy_head = configs["policy_head"]
        self.value_loss_factor = configs["value_loss_factor"]
        self.name = self.finish_name()
        self.model = tf.estimator.Estimator(self.model_fn, self.name)

    def finish_name(self):
        n = "TensorBoardFiles/CM" + self.name
        added = ["VH"] + [str(x) for x in self.value_head]
        added += ["PH"] + [str(x) for x in self.policy_head]
        added = "_".join(added)
        name = n + added + "/"
        return name

    def build_policy_head(self, input_tensor):
        save_model_size = self.current_model_size

        current_layer = input_tensor

        for i, size in enumerate(self.policy_head):
            current_layer = self.hidden_layer(num_hidden=size, input_tensor=current_layer, layer_count=i+10)

        num_output = self.width * (self.width-1)
        output_layer = self.hidden_layer(num_hidden=num_output, input_tensor=current_layer, relu=False)
        output_layer = tf.nn.leaky_relu(output_layer)

        self.current_model_size = save_model_size

        return output_layer

    def build_value_head(self, input_tensor):
        current_layer = input_tensor

        for i, size in enumerate(self.value_head):
            current_layer = self.hidden_layer(num_hidden=size, input_tensor=current_layer, layer_count=i+20)

        predicted_value = self.hidden_layer(num_hidden=1, input_tensor=current_layer, relu=False)
        return predicted_value

    def model_fn(self, features, labels, mode):
        last_layer = self.build_model_beginning(features)

        # build head for value network
        policy_output = self.build_policy_head(last_layer)
        percent_output = tf.nn.softmax(policy_output)

        # build head for policy network
        value_output = self.build_value_head(last_layer)

        combined_output = tf.concat([value_output, percent_output], axis=1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=combined_output)

        labels_value = tf.slice(labels, [0, 0], [-1, 1])
        labels_policy = tf.slice(labels, [0, 1], [-1, -1])

        # Losses
        policy_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_policy, logits=policy_output)
        policy_loss = tf.dtypes.cast(policy_loss, tf.float32)

        value_loss = tf.losses.mean_squared_error(labels=labels_value, predictions=value_output)

        combined_loss = tf.reduce_mean(value_loss + policy_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(combined_loss, global_step=tf.train.get_global_step())

        # acc op for value
        value_acc = tf.metrics.mean_absolute_error(labels=labels_value, predictions=value_output)
        # acc op for policy
        policy_acc = tf.nn.softmax_cross_entropy_with_logits_v2(logits=policy_output, labels=labels_policy)
        policy_acc = tf.metrics.mean(policy_acc)

        estimator_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=combined_output,
            loss=combined_loss,
            train_op=train_op,
            eval_metric_ops={'mean_abs_error': value_acc,
                             'cross_entropy': policy_acc})

        return estimator_specs

    def prepare_data(self, df):
        X = self.prepare_state_data(df)

        y_1 = df.MovesEncoded
        y_1 = np.array([np.array(val, dtype=float) for val in y_1])

        y_2 = df.Value
        y_2 = np.array([np.array([val], dtype=float) for val in y_2])
        y = np.concatenate([y_2, y_1], axis=1)
        return X, y

    def train_df(self, df):
        X, y = self.prepare_data(df)
        self.train(X, y)

    def evaluate_df(self, df):
        X, y = self.prepare_data(df)
        self.evaluate(X, y)


from queue import Queue
from threading import Thread


class EstimatorWrapper:
    def __init__(self, model):
        self.model = model
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

        self.prediction_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.prediction_thread.start()

    def generate_from_queue(self):
        """ Generator which yields items from the input queue.
        This lives within our 'prediction thread'.
        """

        while True:
            yield self.input_queue.get()

    def predict_from_queue(self):
        """ Adds a prediction from the model to the output_queue.
        This lives within our 'prediction thread'.
        Note: estimators accept generators as inputs and return generators as output.
        Here, we are iterating through the output generator, which will be
        populated in lock-step with the input generator.
        """

        for i in self.model.model.predict(input_fn=self.queued_predict_input_fn):
            self.output_queue.put(i)

    def predict_df(self, df):
        # Get predictions dictionary
        X = self.model.prepare_state_data(df)

        for xx in X:
            features = {"x": np.array([xx])}
            self.input_queue.put(features)
            predictions = self.output_queue.get()  # The latest predictions generator
            yield predictions

        return predictions

    def queued_predict_input_fn(self):
        """
        Queued version of the `predict_input_fn` in FlowerClassifier.
        Instead of yielding a dataset from data as a parameter,
        we construct a Dataset from a generator,
        which yields from the input queue.
        """

        # Fetch the inputs from the input queue
        output_types = {'x': tf.float64}
        dataset = tf.data.Dataset.from_generator(self.generate_from_queue, output_types=output_types)

        return dataset


