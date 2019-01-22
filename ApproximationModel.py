import tensorflow as tf


class ApproximationModel:
    def __init__(self):
        self.X = tf.placeholder("float", [None, 10], name="Input")
        self.Y = tf.placeholder("float", [None, 10], name="Y_True")

        self.shared_layer = self.sharded_layer(2, 5, 3)

        self.first_layer = self.hidden_layer(n_input=6, n_hidden=20, name_scope="first_connected_layer",
                                             prev_layer=self.shared_layer)
        self.second_layer = self.hidden_layer(n_input=20, n_hidden=20, name_scope="second_layer",
                                              prev_layer=self.first_layer)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(
                            'TBGraphs/',
                            graph=sess.graph,
                            session=sess)
            writer.add_graph(sess.graph)

            for x in range(4):
                summary = tf.Summary()
                summary.value.add(tag='AverageMoves', simple_value=x)
                writer.add_summary(summary, global_step=x)

            writer.flush()
            writer.close()

    @staticmethod
    def reshape_input(input_matrix):
        return input_matrix.transpose().flatten()

    def sharded_layer(self, num_columns, column_height, num_hidden):
        with tf.name_scope("Shared_layer"):
            init = tf.Variable(tf.random_normal([column_height, num_hidden]))
            weights = tf.get_variable('Shared_weights', initializer=init)

            init = tf.Variable(tf.zeros([num_hidden]))
            biases = tf.get_variable('Shared_biases', initializer=init)

            tensor_list = []
            for x in range(num_columns):
                col = tf.slice(self.X, [0, column_height*x], [1, column_height])
                name = "column_" + str(x)
                layer = tf.nn.relu(tf.matmul(col, weights) + biases)
                next_layer = self.hidden_layer(n_input=num_hidden, n_hidden=3, prev_layer=layer, name_scope=name)
                tensor_list.append(next_layer)

            combination_layer = tf.concat(tensor_list, name="Combination", axis=1)
            return combination_layer

    def hidden_layer(self, n_input, n_hidden, prev_layer, name_scope):
        with tf.name_scope(name_scope):
            bias = tf.Variable(tf.random_normal([n_hidden]))
            weights = tf.Variable(tf.random_normal([n_input, n_hidden]))
            layer = tf.add(tf.matmul(prev_layer, weights), bias)
        return layer

    def train(self, x, y):
        pass

    def predict(self, x, y):
        pass


class PolicyNetwork(ApproximationModel):
    def __init__(self):
        super().__init__()
        pass


class ValueNetwork(ApproximationModel):
    def __init__(self):
        super().__init__()
        pass

test = ApproximationModel()

