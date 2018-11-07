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

    with tf.variable_scope('lstm_layer') as scope:
        state_per_layer_list = tf.unstack(x_lstm, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )
        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        # cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
        # cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        output, state = tf.nn.dynamic_rnn(cell, x_lstm, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, hidden_size)
        lstm_output = tf.reshape(output, [-1, hidden_size])

        lstm_output = lstm_output/tf.reduce_max(tf.abs(lstm_output))


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

    return x, y, y_pred#, global_step, learning_rate


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
