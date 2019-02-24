
import tensorflow as tf
import numpy as np


class RatingModel:
    def __init__(self, data_source = "A+I", concat_type='Additive', conv_type='Both',batch_size = 64, seq_len = 60, learning_rate = 0.00008, epochs = 1, n_channels_user = 100,n_classes = 1, n_channels_audio = 100, n_channels_image = 2048):
        print('Initialize new model')
        self.data_source = data_source
        self.concat_type=concat_type
        self.conv_type=conv_type
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_channels_user = n_channels_user
        self.n_classes = n_classes
        self.n_channels_audio = n_channels_audio
        self.n_channels_image = n_channels_image

    def _create_placeholders(self):
        
        self.inputs_image = tf.placeholder(tf.float32, [None, self.seq_len, self.n_channels_image], name = 'inputs_image')
        self.inputs_audio = tf.placeholder(tf.float32, [None, self.seq_len, self.n_channels_audio], name = 'inputs_audio')
        self.inputs_user = tf.placeholder(tf.float32, [None, self.n_channels_user], name = 'inputs_user')
        
        self.labels_ = tf.placeholder(tf.float32, [None, 1], name = 'labels')
        self.keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        self.learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    def _inference(self):

        # with tf.name_scope("1dconv_layer"):
        if self.conv_type in ['Both','Audio']:
            self.conv1_audio = tf.layers.conv1d(inputs=self.inputs_audio, filters=8, kernel_size=10, strides=1, dilation_rate=1, #tf.linalg.l2_normalize(
                                     padding='same', activation = tf.nn.relu)
            self.max_pool_1_audio = tf.layers.max_pooling1d(inputs=self.conv1_audio, pool_size=2, strides=2, padding='same')

            self.conv2_audio = tf.layers.conv1d(inputs=self.max_pool_1_audio, filters=16, kernel_size=10, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            self.max_pool_2_audio = tf.layers.max_pooling1d(inputs=self.conv2_audio, pool_size=2, strides=2, padding='same')

            self.conv3_audio = tf.layers.conv1d(inputs=self.max_pool_2_audio, filters=32, kernel_size=5, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            self.max_pool_3_audio = tf.layers.max_pooling1d(inputs=self.conv3_audio, pool_size=2, strides=2, padding='same')

            self.conv4_audio = tf.layers.conv1d(inputs=self.max_pool_3_audio, filters=64, kernel_size=2, strides=2, 
                                    padding='same', activation = tf.nn.relu)
            self.max_pool_4_audio = tf.layers.max_pooling1d(inputs=self.conv4_audio, pool_size=2, strides=2, padding='same')
        else:
            self.max_pool_4_audio = tf.reduce_mean(self.inputs_audio, axis=1)
      
        if self.conv_type in ['Both', 'Image']:
            self.conv1_image = tf.layers.conv1d(inputs=self.inputs_image, filters=16, kernel_size=2, strides=1, dilation_rate=1,
                                     padding='same', activation = tf.nn.relu)
            self.max_pool_1_image = tf.layers.max_pooling1d(inputs=self.conv1_image, pool_size=2, strides=2, padding='same')

            self.conv2_image = tf.layers.conv1d(inputs=self.max_pool_1_image, filters=32, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            self.max_pool_2_image = tf.layers.max_pooling1d(inputs=self.conv2_image, pool_size=2, strides=2, padding='same')

            self.conv3_image = tf.layers.conv1d(inputs=self.max_pool_2_image, filters=64, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            self.max_pool_3_image = tf.layers.max_pooling1d(inputs=self.conv3_image, pool_size=2, strides=2, padding='same')

            self.conv4_image = tf.layers.conv1d(inputs=self.max_pool_3_image, filters=128, kernel_size=2, strides=2, 
                                     padding='same', activation = tf.nn.relu)
            self.max_pool_4_image = tf.layers.max_pooling1d(inputs=self.conv4_image, pool_size=2, strides=2, padding='same')
        else:
            self.max_pool_4_image = tf.reduce_mean(self.inputs_image, axis=1)

        # with tf.name_scope("fusion_layer"):
        # Flatten and add dropout
        self.flat_audio = tf.reshape(self.max_pool_4_audio, (-1,int(self.max_pool_4_audio.shape[1])*int(self.max_pool_4_audio.shape[2])))
        #flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)

        self.flat_image = tf.reshape(self.max_pool_4_image, (-1,int(self.max_pool_4_image.shape[1])*int(self.max_pool_4_image.shape[2])))
        #flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)
        
        self.flat_audio = tf.nn.dropout(self.flat_audio, keep_prob=self.keep_prob_)
        self.flat_image = tf.nn.dropout(self.flat_image, keep_prob=self.keep_prob_)
        self.flat_user = tf.nn.dropout(self.inputs_user, keep_prob=self.keep_prob_)

        self.h_v = tf.layers.dense(self.flat_image,
                                  64,
                                  activation=tf.nn.tanh)
        self.h_a = tf.layers.dense(self.flat_audio,
                                  64,
                                  activation=tf.nn.tanh)
        self.h_u = tf.layers.dense(self.flat_user,
                                  64,
                                  activation=tf.nn.tanh)
        self.z_trailer = tf.layers.dense(tf.concat([self.flat_audio, self.flat_image], -1), 
                                  64,
                                  activation=tf.nn.sigmoid)

        self.h_trailer = self.z_trailer * self.h_v + (1-self.z_trailer) * self.h_a

        self.h_trailer = tf.nn.dropout(self.h_trailer, keep_prob=self.keep_prob_)

        self.z = tf.layers.dense(tf.concat([self.h_trailer, self.flat_user], -1),
                                  64,
                                  activation=tf.nn.sigmoid)

        self.h = self.z * self.h_trailer + (1-self.z) * self.h_u

        # self.h = tf.nn.dropout(self.h, keep_prob=self.keep_prob_)

        self.logits = tf.layers.dense(self.h, self.n_classes, name='logits')
    
    def _create_cost(self):

        print('Creating loss... \n')

        # Cost function and optimizer
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels_), name='cost')

    def _create_optimizer_n_beyond(self):

        #self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        #cost = tf.reduce_mean(logits, labels_)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.cost)

        # Accuracy

        self.predicted = tf.nn.sigmoid(self.logits, name='predicted')
        self.correct_pred = tf.equal(tf.round(self.predicted), self.labels_)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')
        self.f1score = tf.contrib.metrics.f1_score(labels=self.labels_,predictions=self.predicted)
        # ROC Curve
        self.gt_ , self.pr_ = self.labels_, self.predicted

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_cost()
        self._create_optimizer_n_beyond()


class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for trailer similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope, embedding_size, frame_size, hidden_units):
        n_hidden=hidden_units
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(x, 30, 1)#tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        #print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
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
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keepdims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keepdims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keepdims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class SiameseLSTM_uni(object):
    """
    A LSTM based deep Siamese network for trailer similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope, embedding_size, frame_size, hidden_units):
        n_hidden=hidden_units
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(x, 30, 1)#tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        #print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
            print(lstm_fw_cell_m)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
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
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keepdims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keepdims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keepdims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


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

