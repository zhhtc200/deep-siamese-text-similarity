import tensorflow as tf


class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, hidden_size):
        n_hidden = hidden_size
        n_layers = 1
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            print(tf.get_variable_scope().name)
            all_fw_cell_list = []
            for layer_id in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(
                    n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    fw_cell, output_keep_prob=dropout)
                all_fw_cell_list.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(all_fw_cell_list,
                                                         state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            print(tf.get_variable_scope().name)
            all_bw_cell_list = []
            for layer_id in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(
                    n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    bw_cell, output_keep_prob=dropout)
                all_bw_cell_list.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(
                all_bw_cell_list, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32
            )
            outputs = outputs[1]
        return tf.reduce_sum(outputs, axis=1)

    def contrastive_loss(self, y, d):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tmp + tmp2

    def __init__(
            self, sequence_length, vocab_size, embedding_size, hidden_units,
            l2_reg_lambda):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True, name="W")
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W,
                                                          self.input_x1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W,
                                                          self.input_x2)

        # Create a convolution + maxpool layer for each filter size
        with tf.variable_scope("output") as scope:
            self.out1 = self.BiRNN(self.embedded_chars1,
                                   self.dropout_keep_prob,
                                   "side1", hidden_units)
            scope.reuse_variables()
            self.out2 = self.BiRNN(self.embedded_chars2,
                                   self.dropout_keep_prob,
                                   "side1", hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(
                tf.square(tf.subtract(self.out1, self.out2)), 1,
                keep_dims=True))
            self.distance = tf.divide(
                self.distance,
                tf.add(
                    tf.sqrt(
                        tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)
                    ),
                    tf.sqrt(
                        tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True)
                    )))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            losses = self.contrastive_loss(self.input_y, self.distance)
            self.loss = tf.losses.compute_weighted_loss(losses)
            L2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss += l2_reg_lambda * L2

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.distance, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
