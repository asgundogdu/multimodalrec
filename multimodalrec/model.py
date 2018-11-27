
import tensorflow as tf
import numpy as np

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for trailer similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope, embedding_size, frame_size, hidden_units):
        n_hidden=hidden_units
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        return outputs[-1]
    
    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
        self, frame_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, frame_size, embedding_size], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, frame_size, embedding_size], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1=self.BiRNN(self.input_x1, self.dropout_keep_prob, "side1", embedding_size, frame_size, hidden_units)
            self.out2=self.BiRNN(self.input_x2, self.dropout_keep_prob, "side2", embedding_size, frame_size, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



import tensorflow as tf

# class Input(object):
#     def __init__(self, batch_size, num_steps, data):
#         self.batch_size = batch_size
#         self.num_steps = num_steps
#         self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#         self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

# def batch_producer(raw_data, batch_size, num_steps):
#     raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

#     data_len = tf.size(raw_data)
#     batch_len = data_len // batch_size
#     data = tf.reshape(raw_data[0: batch_size * batch_len],
#                       [batch_size, batch_len])

#     epoch_size = (batch_len - 1) // num_steps

#     i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
#     x = data[:, i * num_steps:(i + 1) * num_steps]
#     x.set_shape([batch_size, num_steps])
#     y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
#     y.set_shape([batch_size, num_steps])
#     return x, y

def model():
    #image_size = 32
    input_1_dimension = 2048
    hidden_size = 2048
    #num_channels = 3
    input_1_frames = 30
    
    input_2_dimension = 256

    batch_size = 64

    num_layers = 2

    dropout = 0.5

    is_training = True
    
    #num_classes = 10


    # Using name scope to use/understand tensorboard while debugging

    with tf.name_scope('main_parameters'):
        # x = tf.placeholder(tf.float32, shape=[None, image_size * image_size * num_channels], name='Input')
        # y = tf.placeholder(tf.float32, shape=[None, num_classes], name='Output')
        # x_image = tf.reshape(x, [-1, image_size, image_size, num_channels], name='images')
        x_lstm = tf.placeholder(tf.float32, [num_layers, 2, batch_size, input_1_dimension], name='Input_LSTM')
        x_fusion = tf.placeholder(tf.float32, shape=[None, input_2_dimension], name='Input_fusion')

        y = tf.placeholder(tf.float32, shape=[None, 1], name='Output')



    with tf.variable_scope('fusion_layer') as scope:

        lstm_output_fc = tf.layers.dense(inputs=lstm_output, units=512, activation=tf.nn.relu)

        x_fusion_fc = tf.layers.dense(inputs=x_fusion, units=64, activation=tf.nn.relu)

        fused_tensor = tf.concat(concat_dim=1,values=[lstm_output_fc, x_fusion_fc])

        last_layer = tf.layers.dense(inputs=fused_tensor, units=32, activation=tf.nn.sigmoid) # ReLU

    # with tf.variable_scope('fully_connected_layer') as scope:
    #     flat = tf.reshape(drop, [-1, 4 * 4 * 256])

    #     fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
    #     drop = tf.layers.dropout(fc, rate=0.55)

    #     # fc = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
    #     # drop = tf.layers.dropout(fc, rate=0.4)

        y_pred = tf.layers.dense(inputs=last_layer, units=1, activation=tf.nn.softmax, name='regression_layer')

    # y_pred_cls = tf.argmax(softmax, axis=1)

    return x_lstm, x_fusion, y, y_pred#, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-3
    if epoch > 70:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate

# import tensorflow as tf

# # class Input(object):
# #     def __init__(self, batch_size, num_steps, data):
# #         self.batch_size = batch_size
# #         self.num_steps = num_steps
# #         self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
# #         self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

# # def batch_producer(raw_data, batch_size, num_steps):
# #     raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

# #     data_len = tf.size(raw_data)
# #     batch_len = data_len // batch_size
# #     data = tf.reshape(raw_data[0: batch_size * batch_len],
# #                       [batch_size, batch_len])

# #     epoch_size = (batch_len - 1) // num_steps

# #     i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
# #     x = data[:, i * num_steps:(i + 1) * num_steps]
# #     x.set_shape([batch_size, num_steps])
# #     y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
# #     y.set_shape([batch_size, num_steps])
# #     return x, y

# def model():
#     #image_size = 32
#     input_1_dimension = 2048
#     hidden_size = 2048
#     #num_channels = 3
#     input_1_frames = 30
    
#     input_2_dimension = 256

#     batch_size = 64

#     num_layers = 2

#     dropout = 0.5

#     is_training = True
    
#     #num_classes = 10


#     # Using name scope to use/understand tensorboard while debugging

#     with tf.name_scope('main_parameters'):
#         # x = tf.placeholder(tf.float32, shape=[None, image_size * image_size * num_channels], name='Input')
#         # y = tf.placeholder(tf.float32, shape=[None, num_classes], name='Output')
#         # x_image = tf.reshape(x, [-1, image_size, image_size, num_channels], name='images')
#         x_lstm = tf.placeholder(tf.float32, [num_layers, 2, batch_size, input_1_dimension], name='Input_LSTM')
#         x_fusion = tf.placeholder(tf.float32, shape=[None, input_2_dimension], name='Input_fusion')

#         y = tf.placeholder(tf.float32, shape=[None, 1], name='Output')

#     with tf.variable_scope('lstm_layer') as scope:
#         state_per_layer_list = tf.unstack(x_lstm, axis=0)
#         rnn_tuple_state = tuple(
#             [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
#              for idx in range(num_layers)]
#         )
#         # create an LSTM cell to be unrolled
#         cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
#         # add a dropout wrapper if training
#         if is_training and dropout < 1:
#             cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
#         if num_layers > 1:
#             cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

#         # cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
#         # cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

#         output, state = tf.nn.dynamic_rnn(cell, x_lstm, dtype=tf.float32, initial_state=rnn_tuple_state)
#         # reshape to (batch_size * num_steps, hidden_size)
#         lstm_output = tf.reshape(output, [-1, hidden_size])

#         lstm_output = lstm_output/tf.reduce_max(tf.abs(lstm_output))


#     with tf.variable_scope('fusion_layer') as scope:

#         lstm_output_fc = tf.layers.dense(inputs=lstm_output, units=512, activation=tf.nn.relu)

#         x_fusion_fc = tf.layers.dense(inputs=x_fusion, units=64, activation=tf.nn.relu)

#         fused_tensor = tf.concat(concat_dim=1,values=[lstm_output_fc, x_fusion_fc])

#         last_layer = tf.layers.dense(inputs=fused_tensor, units=32, activation=tf.nn.sigmoid) # ReLU

#     # with tf.variable_scope('fully_connected_layer') as scope:
#     #     flat = tf.reshape(drop, [-1, 4 * 4 * 256])

#     #     fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
#     #     drop = tf.layers.dropout(fc, rate=0.55)

#     #     # fc = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
#     #     # drop = tf.layers.dropout(fc, rate=0.4)

#         y_pred = tf.layers.dense(inputs=last_layer, units=1, activation=tf.nn.softmax, name='regression_layer')

#     # y_pred_cls = tf.argmax(softmax, axis=1)

#     return x, y, y_pred#, global_step, learning_rate


# def lr(epoch):
#     learning_rate = 1e-3
#     if epoch > 70:
#         learning_rate *= 0.5e-3
#     elif epoch > 60:
#         learning_rate *= 1e-3
#     elif epoch > 40:
#         learning_rate *= 1e-2
#     elif epoch > 20:
#         learning_rate *= 1e-1
#     return learning_rate
