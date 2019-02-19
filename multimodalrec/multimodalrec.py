# Packages
import os, pickle, sys, random
import numpy as np
from collections import Counter
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# In-built packages
from .trailer import AudioVisualEncoder
from .viewership import BipartiteNetwork, CollaborativeFiltering
from .lossTypes import RMSELossGraph
# from .basicGraph import BasicGraph
sys.path.append("/Users/salihgundogdu/Desktop/gits/multimodalrec/data/")
from data import data_creation as dc


def rating_model(data_source = "A+I", concat_type='Additive', conv_type='Both',batch_size = 64, seq_len = 60, learning_rate = 0.00008, epochs = 1, n_channels_user = 100,n_classes = 1, n_channels_audio = 100, n_channels_image = 2048):
    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_image = tf.placeholder(tf.float32, [None, seq_len, n_channels_image], name = 'inputs_image')
        inputs_audio = tf.placeholder(tf.float32, [None, seq_len, n_channels_audio], name = 'inputs_audio')
        inputs_user = tf.placeholder(tf.float32, [None, n_channels_user], name = 'inputs_user')
        
        labels_ = tf.placeholder(tf.float32, [None, 1], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    # AUDIO
    with graph.as_default():
        if conv_type in ['Both','Audio']:
            # (batch, 30, 2048) --> (batch, 15, 18)
            conv1_audio = tf.layers.conv1d(inputs=inputs_audio, filters=8, kernel_size=10, strides=1, dilation_rate=1, #tf.linalg.l2_normalize(
                                     padding='same', activation = tf.nn.relu)
            max_pool_1_audio = tf.layers.max_pooling1d(inputs=conv1_audio, pool_size=2, strides=2, padding='same')

            # (batch, 15, 18) --> (batch, 8, 36)
            conv2_audio = tf.layers.conv1d(inputs=max_pool_1_audio, filters=16, kernel_size=10, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_2_audio = tf.layers.max_pooling1d(inputs=conv2_audio, pool_size=2, strides=2, padding='same')

            # (batch, 8, 36) --> (batch, 4, 72)
            conv3_audio = tf.layers.conv1d(inputs=max_pool_2_audio, filters=32, kernel_size=5, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_3_audio = tf.layers.max_pooling1d(inputs=conv3_audio, pool_size=2, strides=2, padding='same')

            ##(batch, 16, 72) --> (batch, 2, 144)
            conv4_audio = tf.layers.conv1d(inputs=max_pool_3_audio, filters=64, kernel_size=2, strides=2, 
                                    padding='same', activation = tf.nn.relu)
            max_pool_4_audio = tf.layers.max_pooling1d(inputs=conv4_audio, pool_size=2, strides=2, padding='same')
        else:
            max_pool_4_audio = tf.reduce_mean(inputs_audio, axis=1)
        print(max_pool_4_audio)
    # IMAGE
    with graph.as_default():
        if conv_type in ['Both', 'Image']:
            # (batch, 30, 2048) --> (batch, 15, 18)
            conv1_image = tf.layers.conv1d(inputs=inputs_image, filters=16, kernel_size=2, strides=1, dilation_rate=1,
                                     padding='same', activation = tf.nn.relu)
            max_pool_1_image = tf.layers.max_pooling1d(inputs=conv1_image, pool_size=2, strides=2, padding='same')

            # (batch, 15, 18) --> (batch, 8, 36)
            conv2_image = tf.layers.conv1d(inputs=max_pool_1_image, filters=32, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_2_image = tf.layers.max_pooling1d(inputs=conv2_image, pool_size=2, strides=2, padding='same')

            # (batch, 8, 36) --> (batch, 4, 72)
            conv3_image = tf.layers.conv1d(inputs=max_pool_2_image, filters=64, kernel_size=2, strides=1, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_3_image = tf.layers.max_pooling1d(inputs=conv3_image, pool_size=2, strides=2, padding='same')

            # (batch, 16, 72) --> (batch, 2, 144)
            conv4_image = tf.layers.conv1d(inputs=max_pool_3_image, filters=128, kernel_size=2, strides=2, 
                                     padding='same', activation = tf.nn.relu)
            max_pool_4_image = tf.layers.max_pooling1d(inputs=conv4_image, pool_size=2, strides=2, padding='same')
        else:
            max_pool_4_image = tf.reduce_mean(inputs_image, axis=1)
        print(max_pool_4_image)

    with graph.as_default():
        # Flatten and add dropout
        flat_audio = tf.reshape(max_pool_4_audio, (-1,int(max_pool_4_audio.shape[1])*int(max_pool_4_audio.shape[2])))
        #flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)

        flat_image = tf.reshape(max_pool_4_image, (-1,int(max_pool_4_image.shape[1])*int(max_pool_4_image.shape[2])))
        #flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)
        
        flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)
        flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)
        flat_user = tf.nn.dropout(inputs_user, keep_prob=keep_prob_)

        # Concat Layer

        if concat_type == 'Additive':

            if data_source=="A+I":
                concat_flat = tf.concat([tf.linalg.l2_normalize(flat_user), tf.linalg.l2_normalize(flat_image)], -1) # tf.concat([flat_audio, flat_image, flat_user], -1) #
            elif data_source=="A": # Only Audio
                concat_flat = flat_audio#tf.concat([tf.linalg.l2_normalize(flat_audio)], -1)
            elif data_source=="I": # Only Image
                concat_flat = flat_image#tf.concat([tf.linalg.l2_normalize(flat_image)], -1)
            
            initializer = tf.contrib.layers.xavier_initializer()
            # Predictions
            logits = tf.layers.dense(concat_flat, n_classes, activation=None, kernel_initializer=initializer)

            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_))

            #cost = tf.reduce_mean(logits, labels_)
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

            # Accuracy

            predicted = tf.nn.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), labels_)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            f1score = tf.contrib.metrics.f1_score(labels=labels_,predictions=predicted)
            # ROC Curve
            gt_ , pr_ = labels_, predicted

        elif concat_type == 'Multiplicative':

            h_v = tf.layers.dense(flat_image,
                                  64,
                                  activation=tf.nn.tanh)
            h_a = tf.layers.dense(flat_audio,
                                  64,
                                  activation=tf.nn.tanh)
            h_u = tf.layers.dense(flat_user,
                                  64,
                                  activation=tf.nn.tanh)
            z_trailer = tf.layers.dense(tf.concat([flat_audio, flat_image], -1), 
                                64,
                                activation=tf.nn.sigmoid)
            
            z = tf.layers.dense(tf.concat([flat_audio, flat_image, flat_user], -1),
                                64,
                                activation=tf.nn.sigmoid)


            h = z_trailer * h_v + (1-z_trailer) * h_a
            
            h = z * h + (1-z) * h_u
            # h = tf.nn.dropout(h, keep_prob=keep_prob_)
            print(h)
            logits = tf.layers.dense(h, n_classes)
            
            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_))

            #cost = tf.reduce_mean(logits, labels_)
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

            # Accuracy

            predicted = tf.nn.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), labels_)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            f1score = tf.contrib.metrics.f1_score(labels=labels_,predictions=predicted)
            # ROC Curve
            gt_ , pr_ = labels_, predicted
            
        elif concat_type == 'Multiplicative_Image':

            h_v = tf.layers.dense(flat_image,
                                  64,
                                  activation=tf.nn.tanh)
            h_u = tf.layers.dense(flat_user,
                                  64,
                                  activation=tf.nn.tanh)        
            z = tf.layers.dense(tf.concat([flat_image, flat_user], -1),
                                64,
                                activation=tf.nn.sigmoid)

            h = z * h_v + (1-z) * h_u
            
            print(h)
            logits = tf.layers.dense(h, n_classes)
            
            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_))

            #cost = tf.reduce_mean(logits, labels_)
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

            # Accuracy

            predicted = tf.nn.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), labels_)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            f1score = tf.contrib.metrics.f1_score(labels=labels_,predictions=tf.round(predicted))
            # ROC Curve
            gt_ , pr_ = labels_, predicted

        elif concat_type == 'User':

            concat_flat = flat_user#tf.concat([tf.linalg.l2_normalize(flat_image)], -1)
            
            initializer = tf.contrib.layers.xavier_initializer()
            # Predictions
            logits = tf.layers.dense(concat_flat, n_classes, activation=None, kernel_initializer=initializer)

            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_))

            #cost = tf.reduce_mean(logits, labels_)
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

            # Accuracy

            predicted = tf.nn.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), labels_)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            f1score = tf.contrib.metrics.f1_score(labels=labels_,predictions=predicted)
            # ROC Curve
            gt_ , pr_ = labels_, predicted

    return graph, inputs_image, inputs_audio, inputs_user, labels_, keep_prob_, learning_rate_, \
    logits, cost, optimizer, correct_pred, accuracy, f1score, gt_ , pr_


def genre_model(data_source = "A+I", concat_type='Additive', batch_size = 100, seq_len = 60, learning_rate = 0.0005, epochs = 30, n_classes = 13, n_channels_audio = 100, n_channels_image = 2048):
    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_image = tf.placeholder(tf.float32, [None, seq_len, n_channels_image], name = 'inputs_image')
        inputs_audio = tf.placeholder(tf.float32, [None, seq_len, n_channels_audio], name = 'inputs_audio')
        
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    # AUDIO
    with graph.as_default():
        # (batch, 30, 2048) --> (batch, 15, 18)
        conv1_audio = tf.layers.conv1d(inputs=inputs_audio, filters=8, kernel_size=10, strides=1, dilation_rate=1, #tf.linalg.l2_normalize(
                                 padding='same', activation = tf.nn.relu)
        max_pool_1_audio = tf.layers.max_pooling1d(inputs=conv1_audio, pool_size=2, strides=2, padding='same')
        
        # (batch, 15, 18) --> (batch, 8, 36)
        conv2_audio = tf.layers.conv1d(inputs=max_pool_1_audio, filters=16, kernel_size=10, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_2_audio = tf.layers.max_pooling1d(inputs=conv2_audio, pool_size=2, strides=2, padding='same')
        
        # (batch, 8, 36) --> (batch, 4, 72)
        conv3_audio = tf.layers.conv1d(inputs=max_pool_2_audio, filters=32, kernel_size=5, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_3_audio = tf.layers.max_pooling1d(inputs=conv3_audio, pool_size=2, strides=2, padding='same')
        
        ##(batch, 16, 72) --> (batch, 2, 144)
        conv4_audio = tf.layers.conv1d(inputs=max_pool_3_audio, filters=64, kernel_size=2, strides=2, 
                                padding='same', activation = tf.nn.relu)
        max_pool_4_audio = tf.layers.max_pooling1d(inputs=conv4_audio, pool_size=2, strides=2, padding='same')
        print(max_pool_4_audio)
    # IMAGE
    with graph.as_default():
        # (batch, 30, 2048) --> (batch, 15, 18)
        conv1_image = tf.layers.conv1d(inputs=inputs_image, filters=16, kernel_size=2, strides=1, dilation_rate=1,
                                 padding='same', activation = tf.nn.relu)
        max_pool_1_image = tf.layers.max_pooling1d(inputs=conv1_image, pool_size=2, strides=2, padding='same')
        
        # (batch, 15, 18) --> (batch, 8, 36)
        conv2_image = tf.layers.conv1d(inputs=max_pool_1_image, filters=32, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_2_image = tf.layers.max_pooling1d(inputs=conv2_image, pool_size=2, strides=2, padding='same')
        
        # (batch, 8, 36) --> (batch, 4, 72)
        conv3_image = tf.layers.conv1d(inputs=max_pool_2_image, filters=64, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_3_image = tf.layers.max_pooling1d(inputs=conv3_image, pool_size=2, strides=2, padding='same')
        
        # (batch, 16, 72) --> (batch, 2, 144)
        conv4_image = tf.layers.conv1d(inputs=max_pool_3_image, filters=128, kernel_size=2, strides=2, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_4_image = tf.layers.max_pooling1d(inputs=conv4_image, pool_size=2, strides=2, padding='same')
        print(max_pool_4_image)
    with graph.as_default():
        # Flatten and add dropout
        flat_audio = tf.reshape(max_pool_4_audio, (-1,int(max_pool_4_audio.shape[1])*int(max_pool_4_audio.shape[2])))
        #flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)
        
        flat_image = tf.reshape(max_pool_4_image, (-1,int(max_pool_4_image.shape[1])*int(max_pool_4_image.shape[2])))
        #flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)


        # Concat Layer

        if concat_type == 'Additive':
        
            if data_source=="A+I":
                concat_flat = tf.concat([flat_audio, flat_image], -1) # tf.concat([tf.linalg.l2_normalize(flat_audio), tf.linalg.l2_normalize(flat_image)], -1)
            elif data_source=="A": # Only Audio
                concat_flat = flat_audio#tf.concat([tf.linalg.l2_normalize(flat_audio)], -1)
            elif data_source=="I": # Only Image
                concat_flat = flat_image#tf.concat([tf.linalg.l2_normalize(flat_image)], -1)
            #concat_flat = tf.concat([flat_audio, flat_image], -1)
            
            concat_flat = tf.nn.dropout(concat_flat, keep_prob=keep_prob_)

            
            # Predictions
            logits = tf.layers.dense(concat_flat, n_classes)
            
            final_tensor = tf.nn.sigmoid(logits)
            
            # Cost function and optimizer
            # cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=1)) # tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
            cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
            
            # Accuracy
            correct_prediction = tf.equal(tf.round(final_tensor), tf.round(labels_))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            accuracy2 = tf.reduce_mean(all_labels_true)
            
            # ROC Curve
            gt_ , pr_ = labels_, final_tensor # tf.nn.softmax(final_tensor)

        elif concat_type == 'Multiplicative':

            flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)
            flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)

            h_v = tf.layers.dense(flat_image,
                                  64,
                                  activation=tf.nn.tanh)
            h_a = tf.layers.dense(flat_audio,
                                  64,
                                  activation=tf.nn.tanh)
            z = tf.layers.dense(tf.concat([flat_audio, flat_image], -1), # tf.stack([flat_image, flat_audio], axis=1),
                                64,
                                activation=tf.nn.sigmoid)


            h = z * h_v + (1-z) * h_a
            # h = tf.nn.dropout(h, keep_prob=keep_prob_)
            print(h)
            logits = tf.layers.dense(h, n_classes)

            final_tensor = tf.nn.sigmoid(logits)

            # cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=1))
            cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            # cost1 = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cost)

            # Accuracy
            correct_prediction = tf.equal(tf.round(final_tensor), tf.round(labels_))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            accuracy2 = tf.reduce_mean(all_labels_true)

            # ROC Curve
            gt_ , pr_ = labels_, final_tensor


    return graph, inputs_image, inputs_audio, labels_, keep_prob_, learning_rate_, logits, cost, optimizer, correct_prediction, accuracy, all_labels_true, accuracy2, gt_ , pr_ 


def genre_model_partial_conv(data_source = "A+I", concat_type='Additive', batch_size = 100, seq_len = 60, learning_rate = 0.0005, epochs = 30, n_classes = 13, n_channels_audio = 100, n_channels_image = 2048):
    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_image = tf.placeholder(tf.float32, [None, seq_len, n_channels_image], name = 'inputs_image')
        inputs_audio = tf.placeholder(tf.float32, [None, seq_len, n_channels_audio], name = 'inputs_audio')
        
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    # AUDIO
    with graph.as_default():
        # (batch, 30, 2048) --> (batch, 15, 18)
        conv1_audio = tf.layers.conv1d(inputs=inputs_audio, filters=8, kernel_size=10, strides=1, dilation_rate=1, #tf.linalg.l2_normalize(
                                 padding='same', activation = tf.nn.relu)
        max_pool_1_audio = tf.layers.max_pooling1d(inputs=conv1_audio, pool_size=2, strides=2, padding='same')
        
        # (batch, 15, 18) --> (batch, 8, 36)
        conv2_audio = tf.layers.conv1d(inputs=max_pool_1_audio, filters=16, kernel_size=10, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_2_audio = tf.layers.max_pooling1d(inputs=conv2_audio, pool_size=2, strides=2, padding='same')
        
        # (batch, 8, 36) --> (batch, 4, 72)
        conv3_audio = tf.layers.conv1d(inputs=max_pool_2_audio, filters=32, kernel_size=3, strides=2, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_3_audio = tf.layers.max_pooling1d(inputs=conv3_audio, pool_size=2, strides=2, padding='same')
        
        ##(batch, 16, 72) --> (batch, 2, 144)
        conv4_audio = tf.layers.conv1d(inputs=max_pool_3_audio, filters=64, kernel_size=3, strides=2, 
                                padding='same', activation = tf.nn.relu)
        max_pool_4_audio = tf.layers.max_pooling1d(inputs=conv4_audio, pool_size=2, strides=2, padding='same')
        print(max_pool_4_audio)
    # IMAGE
    with graph.as_default():
        # (batch, 30, 2048) --> (batch, 15, 18)
        flat_image = tf.reduce_mean(inputs_image, axis=1)

    with graph.as_default():
        # Flatten and add dropout
        flat_audio = tf.reshape(max_pool_4_audio, (-1,int(max_pool_4_audio.shape[1])*int(max_pool_4_audio.shape[2])))
        #flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)
        
        # flat_image = tf.reshape(max_pool_4_image, (-1,int(max_pool_4_image.shape[1])*int(max_pool_4_image.shape[2])))
        #flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)
        print('flat image:',flat_image)
        print('flat audio:',flat_audio)

        # Concat Layer

        if concat_type == 'Additive':
        
            if data_source=="A+I":
                concat_flat = tf.concat([flat_audio, flat_image], -1) # tf.concat([tf.linalg.l2_normalize(flat_audio), tf.linalg.l2_normalize(flat_image)], -1)
            elif data_source=="A": # Only Audio
                concat_flat = flat_audio#tf.concat([tf.linalg.l2_normalize(flat_audio)], -1)
            elif data_source=="I": # Only Image
                concat_flat = flat_image#tf.concat([tf.linalg.l2_normalize(flat_image)], -1)
            #concat_flat = tf.concat([flat_audio, flat_image], -1)
            
            concat_flat = tf.nn.dropout(concat_flat, keep_prob=keep_prob_)

            
            # Predictions
            logits = tf.layers.dense(concat_flat, n_classes)
            
            final_tensor = tf.nn.sigmoid(logits)
            
            # Cost function and optimizer
            # cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=1)) # tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
            cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
            
            # Accuracy
            correct_prediction = tf.equal(tf.round(final_tensor), tf.round(labels_))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            accuracy2 = tf.reduce_mean(all_labels_true)
            
            # ROC Curve
            gt_ , pr_ = labels_, final_tensor # tf.nn.softmax(final_tensor)

        elif concat_type == 'Multiplicative':

            flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)
            flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)

            h_v = tf.layers.dense(flat_image,
                                  64,
                                  activation=tf.nn.tanh)
            h_a = tf.layers.dense(flat_audio,
                                  64,
                                  activation=tf.nn.tanh)
            z = tf.layers.dense(tf.concat([flat_audio, flat_image], -1), # tf.stack([flat_image, flat_audio], axis=1),
                                64,
                                activation=tf.nn.sigmoid)


            h = z * h_v + (1-z) * h_a
            # h = tf.nn.dropout(h, keep_prob=keep_prob_)
            print(h)
            logits = tf.layers.dense(h, n_classes)

            final_tensor = tf.nn.sigmoid(logits)

            # cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=1))
            cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            # cost1 = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cost)

            # Accuracy
            correct_prediction = tf.equal(tf.round(final_tensor), tf.round(labels_))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            accuracy2 = tf.reduce_mean(all_labels_true)

            # ROC Curve
            gt_ , pr_ = labels_, final_tensor


    return graph, inputs_image, inputs_audio, labels_, keep_prob_, learning_rate_, logits, cost, optimizer, correct_prediction, accuracy, all_labels_true, accuracy2, gt_ , pr_ 



def genre_model_mean(data_source = "A+I", concat_type='Additive', batch_size = 100, seq_len = 60, learning_rate = 0.0005, epochs = 30, n_classes = 13, n_channels_audio = 100, n_channels_image = 2048):
    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_image = tf.placeholder(tf.float32, [None, seq_len, n_channels_image], name = 'inputs_image')
        inputs_audio = tf.placeholder(tf.float32, [None, seq_len, n_channels_audio], name = 'inputs_audio')
        
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    
    with graph.as_default():
        # Flatten and add dropout
        flat_audio = tf.reduce_mean(inputs_audio, axis=1) #tf.reshape(audio_mean, (-1,int(audio_mean.shape[1])*int(audio_mean.shape[2])))
        #flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)
        
        flat_image = tf.reduce_mean(inputs_image, axis=1) #tf.reshape(max_pool_4_image, (-1,int(max_pool_4_image.shape[1])*int(max_pool_4_image.shape[2])))
        #flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)


        # Concat Layer

        if concat_type == 'Additive':
        
            if data_source=="A+I":
                concat_flat = tf.concat([flat_audio, flat_image], -1) # tf.concat([tf.linalg.l2_normalize(flat_audio), tf.linalg.l2_normalize(flat_image)], -1)
            elif data_source=="A": # Only Audio
                concat_flat = flat_audio#tf.concat([tf.linalg.l2_normalize(flat_audio)], -1)
            elif data_source=="I": # Only Image
                concat_flat = flat_image#tf.concat([tf.linalg.l2_normalize(flat_image)], -1)
            #concat_flat = tf.concat([flat_audio, flat_image], -1)
            
            concat_flat = tf.nn.dropout(concat_flat, keep_prob=keep_prob_)

            
            # Predictions
            logits = tf.layers.dense(concat_flat, n_classes)
            
            final_tensor = tf.nn.sigmoid(logits)
            
            # Cost function and optimizer
            # cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=1)) # tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
            cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
            
            # Accuracy
            correct_prediction = tf.equal(tf.round(final_tensor), tf.round(labels_))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            accuracy2 = tf.reduce_mean(all_labels_true)
            
            # ROC Curve
            gt_ , pr_ = labels_, final_tensor # tf.nn.softmax(final_tensor)

        elif concat_type == 'Multiplicative':

            flat_image = tf.nn.dropout(flat_image, keep_prob=keep_prob_)
            flat_audio = tf.nn.dropout(flat_audio, keep_prob=keep_prob_)

            h_v = tf.layers.dense(flat_image,
                                  64,
                                  activation=tf.nn.tanh)
            h_a = tf.layers.dense(flat_audio,
                                  64,
                                  activation=tf.nn.tanh)
            z = tf.layers.dense(tf.concat([flat_audio, flat_image], -1), # tf.stack([flat_image, flat_audio], axis=1),
                                64,
                                activation=tf.nn.sigmoid)


            h = z * h_v + (1-z) * h_a
            # h = tf.nn.dropout(h, keep_prob=keep_prob_)
            print(h)
            logits = tf.layers.dense(h, n_classes)

            final_tensor = tf.nn.sigmoid(logits)

            # cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=1))
            cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            # cost1 = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=logits)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cost)

            # Accuracy
            correct_prediction = tf.equal(tf.round(final_tensor), tf.round(labels_))
            
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            accuracy2 = tf.reduce_mean(all_labels_true)

            # ROC Curve
            gt_ , pr_ = labels_, final_tensor


    return graph, inputs_image, inputs_audio, labels_, keep_prob_, learning_rate_, logits, cost, optimizer, correct_prediction, accuracy, all_labels_true, accuracy2, gt_ , pr_ 



def audio_batch_generator(df, batchsize=32):
    movies = np.split(df, df.movie_id.nunique())
    for batch_i in range(len(movies)//batchsize):        
        temp_mov = movies[batch_i*batchsize:batch_i*batchsize+batchsize]
        batch_x, batch_y = [], [] 
        for mov in temp_mov:
            batch_x.append(mov.iloc[:,:-2].values)
            #print(mov.iloc[:,-2].values[0])
            batch_y.append(np.array(mov.iloc[:,-2].values[0]))
        #print(batch_i)
        yield np.array(batch_x)/100., np.array(batch_y)

        
def audio_val_batch_generator(df, batchsize=1):
    # if validation:
    #     df = df.iloc[:(df.movie_id.nunique()//2)*60,:].copy()
    # else:
    #     df = df.iloc[(df.movie_id.nunique()//2)*60:,:].copy()
    movies = np.split(df, df.movie_id.nunique())
    batch_x, batch_y = [], [] 
    for batch_i in range(len(movies)//batchsize):        
        temp_mov = movies[batch_i*batchsize:batch_i*batchsize+batchsize]
        
        for mov in temp_mov:
            batch_x.append(mov.iloc[:,:-2].values)
            #print(mov.iloc[:,-2].values[0])
            batch_y.append(np.array(mov.iloc[:,-2].values[0]))
        #print(batch_i)
    return np.array(batch_x)/100., np.array(batch_y)


def vis_batch_generator(df, batchsize=32):
    movies = np.split(df, df.movie_id.nunique())
    for batch_i in range(len(movies)//batchsize):        
        temp_mov = movies[batch_i*batchsize:batch_i*batchsize+batchsize]
        batch_x, batch_y = [], [] 
        for mov in temp_mov:
            batch_x.append(mov.iloc[:,:-2].values)
            #print(mov.iloc[:,-2].values[0])
            batch_y.append(np.array(mov.iloc[:,-2].values[0]))
        #print(batch_i)
        yield np.array(batch_x), np.array(batch_y)
        

def vis_val_batch_generator(df, batchsize=1):
    # if validation:
    #     df = df.iloc[:(df.movie_id.nunique()//2)*60,:].copy()
    # else:
    #     df = df.iloc[(df.movie_id.nunique()//2)*60:,:].copy()
    movies = np.split(df, df.movie_id.nunique())
    batch_x, batch_y = [], [] 
    for batch_i in range(len(movies)//batchsize):        
        temp_mov = movies[batch_i*batchsize:batch_i*batchsize+batchsize]
        
        for mov in temp_mov:
            batch_x.append(mov.iloc[:,:-2].values)
            # print(mov.iloc[:,-2].values[0])
            batch_y.append(np.array(mov.iloc[:,-2].values[0]))
        #print(batch_i)
    return np.array(batch_x), np.array(batch_y)



def batch_generator(df, audio_features, visual_features, user_latent_traninig, batchsize=32):
    for batch in np.array_split(df, batchsize): 
        batch_x_audio, batch_x_image, batch_x_user, batch_y = [], [], [], [] 
        for ind, row in batch.iterrows():
            batch_x_audio.append(audio_features[row['Movie']])
            batch_x_image.append(visual_features[row['Movie']])
            batch_x_user.append(user_latent_traninig[row['User']])
            batch_y.append(int(row['Rating']>3.5))
    
        yield np.array(batch_x_audio)/100., np.array(batch_x_image), np.array(batch_x_user), np.array(batch_y)

def val_batch_generator(df, audio_features, visual_features, user_latent_traninig, batchsize=1):
    for batch in np.array_split(df, batchsize): 
        batch_x_audio, batch_x_image, batch_x_user, batch_y = [], [], [], [] 
        for ind, row in (batch.iterrows()):
            batch_x_audio.append(audio_features[row['Movie']])
            batch_x_image.append(visual_features[row['Movie']])
            batch_x_user.append(user_latent_traninig[row['User']])
            batch_y.append(int(row['Rating']>3.5))
    
        return np.array(batch_x_audio)/100., np.array(batch_x_image), np.array(batch_x_user), np.array(batch_y)


class MultimodalRec(object):
    """DocString"""
    def __init__(self,
                audio_encoding='MFCC', # Type of Encodings for Audio Data (STRING)
                visual_encoding='PreTrained_CNN', # Type of Encodings for Visual Data (STRING)
                viewership_encoding='Node2Vec', # Type of Encodings for Viewership Data (STRING) 'Node2Vec' or 'LatentFactor'
                n_visual=128, # Dimension of the Visual Representation Vector (INT)
                n_audial=128, # Dimension of the Visual Representation Vector (INT)
                video_processor=None,#AudioVisualEncoder(), # Encoder to Extract AudioVisual Representations of Given Movie Trailers (OBJ)
                system='SVM' # 'Basic' To compare different architectures

                #user_item_network=BipartiteNetwork(), # U-I Adjacency Graph to compute User and Item Representation (OBJ)

                ):

        # ADD check version

        # ADD check args*

        self.audio_encoding = audio_encoding
        self.visual_encoding = visual_encoding
        self.viewership_encoding = viewership_encoding
        self.n_visual = n_visual
        self.n_audial = n_audial
        self.video_processor=video_processor
        self.system = system
        if viewership_encoding == 'Node2Vec':
            self.user_item_network_training = None#BipartiteNetwork()
            self.user_item_network_test = None
        elif viewership_encoding == 'LatentFactor':
            self.user_item_network_training = None#CollaborativeFiltering()
            self.user_item_network_test = None

           # self.data_generator = DataGenerator()
        self.user_latent_traninig=None
        self.movie_latent_traninig=None

        self.user_latent_test=None
        self.movie_latent_test=None

        self.visual_features=None
        self.audio_features=None

        self.df_image=None
        self.df_image_val=None
        self.df_image_test=None
        self.df_audio=None
        self.df_audio_val=None
        self.df_audio_test=None
        self.df_user=None

        self.data=None

        self.processed_data=None


        # ADD Tensorflow Graph Naming

    def organize_multimodal_data(self, load=False, dataset=1, trailer_directory='data/', sequence_dir='data/', audio_directory='/Volumes/TOSHIBA EXT/audio_samples10M', no_audio=False):

        if dataset==1:
            directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-1m/ratings.dat'
            all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
            pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/pickles/'
            data = dc.get_movielens_1M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)
        elif dataset==10:
            # directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-1m/ratings.dat'
            # all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
            # pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/pickles/'
            # data1M = dc.get_movielens_1M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)
            
            # directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-10m/ratings.dat'
            # data = dc.get_movielens_10M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)

            directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-10m/ratings.dat'
            all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
            pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data10/pickles/'
            # data1M = dc.get_movielens_1M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)

            directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-10m/ratings.dat'
            data = dc.get_movielens_10M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)


            # Applying same movies (no need to process all images again)
            #data['test'] = data['training'][data['training'].Movie.isin(data1M['test'].Movie.unique().tolist())]
            #data['training'] = data['training'][data['training'].Movie.isin(data1M['training'].Movie.unique().tolist())]
            #data['Titles'] = data1M['Titles']
            print(data['training'].shape)
        elif dataset==20:
            raise ValueError('20M dataset is not yet implemented')
        else:
            raise ValueError('dataset selection is not valid! Use 1, 10 or 20 as int type')

        # Get Latent Factors Training
        print('Training User-Movie Latent Factors are extracting...')
        print(data['training'].shape)
        self.user_item_network_training = CollaborativeFiltering(data['training']) # 0 Threshold trim
        user_latent_traninig, movie_latent_traninig, sigma = self.user_item_network_training.compute_latent_factors(algorithm='SVD', k=50)
        self.user_latent_traninig = user_latent_traninig
        self.movie_latent_traninig = movie_latent_traninig
        print('Done.')

        #

        # Get Representations of Trailer Frames
        print('Visual Representations are extracting...')
        self.video_processor = AudioVisualEncoder()
        sequences = self.video_processor.extract_visual_features(_dir_=trailer_directory,load=load, seq_dir=sequence_dir)
        self.visual_features = sequences
        print('Done.')

        if not no_audio:
            # Get Representations of Trailer Audio
            print('Audio Representations are extracting...')
            sequences_audio = self.video_processor.extract_audio_features(_dir_=audio_directory)
            self.audio_features = sequences_audio
            print('Done.')

        data['Titles']['MovieGenre'] = data['Titles'].MovieGenre.apply(lambda x: x.split('|'))
        self.data = data


    def create_genreclassification_dataset(self, rand_state=123, sequence_lenght=60):

        # Create Training and Validation Set
        unique_movies_train = self.data['training'].Movie.unique().tolist()
        mov_to_rm = []
        for id_ in self.data['training'].Movie.unique().tolist() + self.data['test'].Movie.unique().tolist():
            if not id_ in list(self.visual_features.keys()):
                # print(id_)
                continue
            if self.visual_features[id_].shape[0] < 60:
                mov_to_rm.append(id_)
        unique_movies_train = list(set(unique_movies_train) - set(mov_to_rm))

        unique_movies_test = self.data['test'].Movie.unique().tolist()
        mov_to_rm = []
        for id_ in self.data['training'].Movie.unique().tolist()+self.data['test'].Movie.unique().tolist():
            if not id_ in list(self.visual_features.keys()):
                # print(id_)
                continue
            if self.visual_features[id_].shape[0] < 60:
                mov_to_rm.append(id_)
        unique_movies_test = list(set(unique_movies_test) - set(mov_to_rm))

        unique_movs = unique_movies_train + unique_movies_test
        random.seed(rand_state)
        random_unique_movies = random.sample(unique_movs,1000)
        unique_movies_train = list(set(unique_movs) - set(random_unique_movies))
        unique_movies_test = random_unique_movies[:500]
        unique_movies_validation = random_unique_movies[500:]
        print('Training Sample:',len(unique_movies_train))
        print('Validation Sample:',len(unique_movies_validation))
        print('Test Sample:',len(unique_movies_test))

        random.seed(123)
        # Create One-hot Genre Vector
        print("Prepearing one-hot multi-label vectors...")

        all_genres = list(set([item for sublist in self.data['Titles'].MovieGenre.tolist() for item in sublist]))
        genre_counts = dict(Counter([item for sublist in self.data['Titles'].MovieGenre.tolist() for item in sublist]))
        all_genres = [genre for genre in all_genres if genre_counts[genre]>500]

        genre_vectors = []
        for ind, row in self.data['Titles'].iterrows():
            multiple_genre_output = []
            for genre in all_genres:
                if genre in row['MovieGenre']:
                    multiple_genre_output.append(1)
                else:
                    multiple_genre_output.append(0) 
            genre_vectors.append(multiple_genre_output)
        len(genre_vectors)

        self.data['Titles']['Output'] = genre_vectors
        titles_df = self.data['Titles']

        # Create Feature Dataframes
        print("Prepearing Audio Features...")

        df_audio = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_train, total=len(unique_movies_train), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), sequence_lenght)),:]) 
                    df_tmp['movie_genre'] = [titles_df[titles_df.MovieID==sample].Output.values[0]] * df_tmp.shape[0]
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                    # df_audio = pd.concat([df_audio, df_tmp],axis=0)
                else:
                    # print(sample)
                    pass
        df_audio = pd.concat(df_list, axis=0)
        del df_list
        # print(df_audio.shape)
        
        df_audio_val = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_validation, total=len(unique_movies_validation), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), sequence_lenght)),:]) 
                    df_tmp['movie_genre'] = [titles_df[titles_df.MovieID==sample].Output.values[0]] * df_tmp.shape[0]
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    # print(sample)
                    pass
        df_audio_val = pd.concat(df_list, axis=0)
        del df_list

        df_audio_test = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_test, total=len(unique_movies_test), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), sequence_lenght)),:]) 
                    df_tmp['movie_genre'] = [titles_df[titles_df.MovieID==sample].Output.values[0]] * df_tmp.shape[0]
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                    # df_audio_test = pd.concat([df_audio_test, df_tmp],axis=0)
                else:
                    # print(sample)
                    pass
        df_audio_test = pd.concat(df_list, axis=0)
        del df_list
        # print(df_audio_test.shape)

        self.df_audio = df_audio
        self.df_audio_val = df_audio_val
        self.df_audio_test = df_audio_test

        print("Prepearing Visual Features...")

        df_image = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_train, total=len(unique_movies_train), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), sequence_lenght)),:]) 
                    df_tmp['movie_genre'] = [titles_df[titles_df.MovieID==sample].Output.values[0]]*df_tmp.shape[0]
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    # print(sample)
                    pass
        df_image = pd.concat(df_list, axis=0)
        del df_list
        # print(df_image.shape)'

        df_image_val = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_validation, total=len(unique_movies_validation), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), sequence_lenght)),:])  
                    df_tmp['movie_genre'] = [titles_df[titles_df.MovieID==sample].Output.values[0]]*df_tmp.shape[0]
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    pass
        df_image_val = pd.concat(df_list, axis=0)
        del df_list

        df_image_test = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_test, total=len(unique_movies_test), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), sequence_lenght)),:])  
                    df_tmp['movie_genre'] = [titles_df[titles_df.MovieID==sample].Output.values[0]]*df_tmp.shape[0]
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)#df_image_test = pd.concat([df_image_test, df_tmp],axis=0)
                else:
                    # print(sample)
                    pass
        df_image_test = pd.concat(df_list, axis=0)
        del df_list
        # print(df_image_test.shape)

        df_image = df_image[df_image.movie_id.isin(df_audio.movie_id.unique().tolist())]
        df_image_val = df_image_val[df_image_val.movie_id.isin(df_audio_val.movie_id.unique().tolist())]
        df_image_test = df_image_test[df_image_test.movie_id.isin(df_audio_test.movie_id.unique().tolist())]

        self.df_image = df_image
        self.df_image_val = df_image_val
        self.df_image_test = df_image_test


    def eval_genre_classification(self, data_source = "A+I", batch_size = 100, learning_rate = 0.0005, epochs = 30, save_dir = "./model/audio/"):
        graph, inputs_image, inputs_audio, labels_, keep_prob_, \
        learning_rate_, logits, cost, optimizer, correct_prediction, \
        accuracy, all_labels_true, accuracy2, gt_ , pr_ = genre_model(data_source = data_source, batch_size = batch_size, 
                                                                      learning_rate = learning_rate, epochs = epochs)

        # with graph.as_default():
            
        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            # sess = tf.Session()

            try:
                print("\nTrying to restore last checkpoint ...")
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
                saver.restore(sess, save_path=last_chk_path)
                print("Restored checkpoint from:", last_chk_path)
            except ValueError:
                print("\nFailed to restore checkpoint. Initializing variables instead.")
                sess.run(tf.global_variables_initializer())

            print("Creating Static Test Set")
            x_t_aud, y_t_aud = audio_val_batch_generator(self.df_audio_val, batchsize=1)
            x_t_vis, y_t_vis = vis_val_batch_generator(self.df_image_val, batchsize=1)
            y_t = y_t_vis


            feed = {inputs_image: x_t_vis, inputs_audio : x_t_aud,  
                                    labels_ : y_t, keep_prob_ : 1.0}  

            loss_t, acc_t, y_gt_t, y_score_t, acc_2_t = sess.run([cost, accuracy, gt_, pr_, accuracy2], feed_dict = feed)

            print(
                  "Test loss: {:6f}".format(loss_t),
                  "Test acc2: {:6f}".format(acc_2_t),
                  "Test acc: {:.6f}".format(acc_t))

            print(y_gt_t[0])
            print(y_score_t[0])

        return loss_t, acc_t, y_gt_t, y_score_t, acc_2_t


    def train_genre_classification(self, data_source = "A+I", concat_type = 'Additive', batch_size = 100, learning_rate = 0.0005, epochs = 30, save_dir = "./model/audio/", mean_flag=False):

        if mean_flag == 'mean':
            graph, inputs_image, inputs_audio, labels_, keep_prob_, \
            learning_rate_, logits, cost, optimizer, correct_prediction, \
            accuracy, all_labels_true, accuracy2, gt_ , pr_ = genre_model_mean(data_source = data_source, concat_type = concat_type,batch_size = batch_size, 
                                                                      learning_rate = learning_rate, epochs = epochs)
        
        elif mean_flag == 'partial':
            graph, inputs_image, inputs_audio, labels_, keep_prob_, \
            learning_rate_, logits, cost, optimizer, correct_prediction, \
            accuracy, all_labels_true, accuracy2, gt_ , pr_ = genre_model_partial_conv(data_source = data_source, concat_type = concat_type,batch_size = batch_size, 
                                                                      learning_rate = learning_rate, epochs = epochs)

        else:
            graph, inputs_image, inputs_audio, labels_, keep_prob_, \
            learning_rate_, logits, cost, optimizer, correct_prediction, \
            accuracy, all_labels_true, accuracy2, gt_ , pr_ = genre_model(data_source = data_source, concat_type = concat_type,batch_size = batch_size, 
                                                                          learning_rate = learning_rate, epochs = epochs)

        validation_gt = []
        validation_pr = []

        validation_acc = []
        validation_acc2 = []
        validation_loss = []

        test_gt = []
        test_pr = []

        test_acc = []
        test_acc2 = []
        test_loss = []

        train_acc2 = []
        train_acc = []
        train_loss = []

        # For Saving the Model
        with graph.as_default():
            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            # saver = tf.train.Saver()
            # sess = tf.Session()
            # train_writer = tf.summary.FileWriter(save_dir, sess.graph)
        
        # Creating Static Validation Set
        x_v_aud, y_v_aud = audio_val_batch_generator(self.df_audio_val, batchsize=1)
        x_v_vis, y_v_vis = vis_val_batch_generator(self.df_image_val, batchsize=1)
        y_v = y_v_vis
        # print(x_v_aud.shape)
        # print(x_v_aud[0,:])
        # print(x_v_aud[-1,:])
        # print()
        
        # Creating Static Test Set
        x_t_aud, y_t_aud = audio_val_batch_generator(self.df_audio_test, batchsize=1)
        x_t_vis, y_t_vis = vis_val_batch_generator(self.df_image_test, batchsize=1)
        y_t = y_t_vis
        # print(x_t_aud.shape)
        # print(x_t_aud[0,:])
        # print(x_t_aud[-1,:])
        # print()

        print('Training is starting...')

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            val_iteration = 0
            global_loss = 100
            # Loop over epochs
            for e in range(epochs):
                # Loop over batches
                for (x_aud,y_aud), (x_vis, y_vis) in zip(audio_batch_generator(self.df_audio, batch_size), vis_batch_generator(self.df_image, batch_size)):
                    assert np.mean(y_aud) == np.mean(y_vis)
                    y = y_vis
                    
                    # Feed dictionary
                    feed = {inputs_image : x_vis, inputs_audio : x_aud, 
                            labels_ : y, keep_prob_ : 0.3, learning_rate_ : learning_rate}
                    
                    # Loss
                    loss, _ , acc, acc2 = sess.run([cost, optimizer, accuracy, accuracy2], feed_dict = feed)
                    train_acc.append(acc)
                    train_acc2.append(acc2)
                    train_loss.append(loss)
                    
                    # Print at each 5 iters
                    if (iteration % 40 == 0):
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))
                        
                    # Compute validation loss at every 20 iterations
                    if (iteration%20 == 0):                
                        val_acc_ = []
                        val_acc2_ =[]
                        val_loss_ = []

                        tst_acc_ = []
                        tst_acc2_ =[]
                        tst_loss_ = []
                        
                        val_batch_gt = np.empty((0,13), float)
                        val_batch_score = np.empty((0,13), float)

                        tst_batch_gt = np.empty((0,13), float)
                        tst_batch_score = np.empty((0,13), float)
                        
                        # x_v_aud, y_v_aud = audio_val_batch_generator(self.df_audio_test, batchsize=1)
                        # x_v_vis, y_v_vis = vis_val_batch_generator(self.df_image_test, batchsize=1)
                        # y_v = y_v_vis

                        # Feed
                        feed = {inputs_image: x_v_vis, inputs_audio : x_v_aud,  
                                labels_ : y_v, keep_prob_ : 1.0}  

                        # Loss
                        loss_v, acc_v, y_gt_v, y_score_v, acc_2 = sess.run([cost, accuracy, gt_, pr_, accuracy2], feed_dict = feed)           
                        val_acc_.append(acc_v)
                        val_acc2_.append(acc_2)
                        val_loss_.append(loss_v)
                        
                        val_batch_gt = np.append(val_batch_gt, y_gt_v, axis=0)
                        val_batch_score = np.append(val_batch_score, y_score_v, axis=0)
                        
                        if (iteration%40 == 0):       
                            # Print info
                            print("Epoch: {}/{}".format(e, epochs),
                                  "Iteration: {:d}".format(iteration),
                                  "Validation loss: {:6f}".format(np.mean(val_loss_)),
                                  "Validation acc: {:.6f}".format(np.mean(val_acc_)),
                                  "Validation acc2: {:6f}".format(np.mean(val_acc2_)))

                        if iteration>=40:
                            # print("Validation loss",val_loss_)
                            # print("Validation loss len",len(validation_loss))
                            if iteration<= 40:
                                global_loss = validation_loss[val_iteration-1]
                            if global_loss > np.mean(val_loss_):
                                saver.save(sess, save_path=save_dir, global_step=iteration)
                                mes = "This iteration receive better loss: {:.2f} < {:.2f}. Saving session..."
                                print(mes.format(np.mean(val_loss_), global_loss))
                                global_loss = np.mean(val_loss_)
                        # Store
                        validation_acc.append(np.mean(val_acc_))
                        validation_acc2.append(np.mean(val_acc2_))
                        validation_loss.append(np.mean(val_loss_))
                        
                        validation_gt.append(val_batch_gt)
                        validation_pr.append(val_batch_score)

                        val_iteration += 1

                        # x_t_aud, y_t_aud = audio_val_batch_generator(self.df_audio_test, batchsize=1, validation=False)
                        # x_t_vis, y_t_vis = vis_val_batch_generator(self.df_image_test, batchsize=1, validation=False)
                        # y_t = y_t_vis

                        # Feed
                        feed = {inputs_image: x_t_vis, inputs_audio : x_t_aud,  
                                labels_ : y_t, keep_prob_ : 1.0}  

                        # Loss
                        loss_t, acc_t, y_gt_t, y_score_t, acc_2_t = sess.run([cost, accuracy, gt_, pr_, accuracy2], feed_dict = feed)           
                        tst_acc_.append(acc_t)
                        tst_acc2_.append(acc_2_t)
                        tst_loss_.append(loss_t)
                        
                        tst_batch_gt = np.append(tst_batch_gt, y_gt_t, axis=0)
                        tst_batch_score = np.append(tst_batch_score, y_score_t, axis=0)
                        
                        # Store
                        test_acc.append(np.mean(tst_acc_))
                        test_acc2.append(np.mean(tst_acc2_))
                        test_loss.append(np.mean(tst_loss_))
                        
                        test_gt.append(tst_batch_gt)
                        test_pr.append(tst_batch_score)
                    
                    # Iterate 
                    iteration += 1


        return validation_gt, validation_pr, validation_acc, validation_acc2, validation_loss, train_acc, train_loss, train_acc2, test_gt, test_pr, test_acc, test_acc2, test_loss


    def train_rating_prediction(self, train=None, val=None, test=None, data_source="A+I", concat_type='Additive', conv_type='Both',
                batch_size = 64, seq_len = 60, max_iter=4000,
                learning_rate = 0.00008, epochs = 1, n_channels_user = 100,
                n_classes = 1, n_channels_audio = 100, n_channels_image = 2048):

        graph, inputs_image, inputs_audio, inputs_user, labels_, keep_prob_, learning_rate_, \
        logits, cost, optimizer, correct_pred, accuracy, f1score, gt_ , pr_ = rating_model(data_source = data_source, concat_type=concat_type, 
                                                                                        conv_type=conv_type,
                                                                                        batch_size = batch_size, seq_len = seq_len, 
                                                                                        learning_rate = learning_rate, epochs = epochs, n_channels_user = n_channels_user,
                                                                                        n_classes = n_classes, n_channels_audio = n_channels_audio, 
                                                                                        n_channels_image = n_channels_image)
        validation_gt = []
        validation_pr = []

        validation_acc = []
        validation_loss = []

        test_gt = []
        test_pr = []

        test_acc = []
        test_loss = []

        # train_auc = []
        train_acc = []
        train_loss = []

        with graph.as_default():
            saver = tf.train.Saver()

        print('Train, Validation, Test shapes:', train.shape,val.shape,test.shape) # (1254251, 4) (43867, 4) (15798, 4) year_start=2001

        validation_acc = []
        validation_loss = []
        validation_f1 = []

        train_acc = []
        train_loss = []

        with graph.as_default():
            saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            iteration = 1
            val_iteration = 0
            global_loss = 100
            # Loop over epochs
            for e in range(epochs):
                # Loop over batches
                for audio, image, user, y in batch_generator(train, self.df_audio, self.df_image, self.user_latent_traninig, batchsize=train.shape[0]//batch_size):
                    # assert np.mean(y_aud) == np.mean(y_vis)
                    
                    y = np.expand_dims(y, 1)
                    
            
                    # Feed dictionary
                    feed = {inputs_image : image, inputs_audio : audio, inputs_user : user, 
                            labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
                    
                    # Loss
                    loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
                    train_acc.append(acc)
                    train_loss.append(loss)
                    
                    # Print at each 40 iters
                    if (iteration % 20 == 0):
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))
                        
                    # Compute validation loss at every 10 iterations
                    if (iteration%100 == 0) or (iteration == 1): 
                        # VALIDATION SET
                        val_acc_ = []
                        val_loss_ = []
                        val_f1_ = []
                        
                        val_batch_gt = np.empty((0,1), float)
                        val_batch_score = np.empty((0,1), float)
                        
                        for audio_val, image_val, user_val, y_val_ in batch_generator(val, self.df_audio_val, self.df_image_val, self.user_latent_traninig, batchsize=val.shape[0]//batch_size):
                            # Feed
                            y_val = np.expand_dims(y_val_, 1)
                            feed = {inputs_image : image_val, inputs_audio : audio_val, inputs_user : user_val, 
                                    labels_ : y_val, keep_prob_ : 1.0}  

                            # Loss
                            loss_v, acc_v, logs, labs, _gt_, _pr_, _f1_ = sess.run([cost, accuracy,logits,labels_,gt_ , pr_, f1score], feed_dict = feed)                    
                            val_acc_.append(acc_v)
                            val_loss_.append(loss_v)
                            val_f1_.append(_f1_[0])

                            val_batch_gt = np.append(val_batch_gt, _gt_, axis=0)
                            val_batch_score = np.append(val_batch_score, _pr_, axis=0)
                            
                            # Print info
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {:d}".format(iteration),
                              "Validation loss: {:6f}".format(np.mean(val_loss_)),
                              "Validation acc: {:.6f}".format(np.mean(val_acc_)),
                              "Validation F1 score: {:.6f}".format(np.mean(val_f1_[0])))
                        
                        if iteration>=100:
                            if iteration<= 100:
                                global_loss = validation_loss[val_iteration-1]
                            if global_loss > np.mean(val_loss_):
                                # saver.save(sess, save_path=save_dir, global_step=iteration)
                                mes = "This iteration receive better loss: {:.3f} < {:.3f}. Saving session..."
                                print(mes.format(np.mean(val_loss_), global_loss))
                                global_loss = np.mean(val_loss_)
                        
                        # Store
                        validation_acc.append(np.mean(val_acc_))
                        validation_loss.append(np.mean(val_loss_))
                        validation_f1.append(np.mean(val_f1_))
                        
                        validation_gt.append(val_batch_gt)
                        validation_pr.append(val_batch_score)
                        
                        
                        # TEST SET
                        test_acc_ = []
                        test_loss_ = []
                        test_batch_gt = np.empty((0,1), float)
                        test_batch_score = np.empty((0,1), float)
                        for audio_test, image_test, user_test, y_test_ in batch_generator(test, self.df_audio_test, self.df_image_test, self.user_latent_traninig, batchsize=test.shape[0]//batch_size):
                            # Feed
                            y_test = np.expand_dims(y_test_, 1)
                            feed = {inputs_image : image_test, inputs_audio : audio_test, inputs_user : user_test, 
                                    labels_ : y_test, keep_prob_ : 1.0}  

                            # Loss
                            loss_t, acc_t, logs_t, labs_t, _gt_t, _pr_t = sess.run([cost, accuracy,logits,labels_,gt_ , pr_], feed_dict = feed)                    
                            test_acc_.append(acc_t)
                            test_loss_.append(loss_t)

                            test_batch_gt = np.append(test_batch_gt, _gt_t, axis=0)
                            test_batch_score = np.append(test_batch_score, _pr_t, axis=0)
                            
                        # Store test
                        test_acc.append(np.mean(test_acc_))
                        test_loss.append(np.mean(test_loss_))
                        
                        test_gt.append(test_batch_gt)
                        test_pr.append(test_batch_score)

                    
                    # Iterate 
                    iteration += 1
                    if iteration >= max_iter: break

        return validation_gt, validation_pr, validation_acc, validation_loss, train_acc, train_loss, test_gt, test_pr, test_acc, test_loss


    def assign_all_zero_genre_classification(self, data_source = "Zero", batch_size = 100, learning_rate = 0.0005, 
                                             epochs = 30, save_dir = "./model/audio/"):

        
        # Creating Static Validation Set
        x_v_aud, y_v_aud = audio_val_batch_generator(self.df_audio_val, batchsize=1)
        x_v_vis, y_v_vis = vis_val_batch_generator(self.df_image_val, batchsize=1)
        y_v = y_v_vis
        # print(x_v_aud.shape)
        # print(x_v_aud[0,:])
        # print(x_v_aud[-1,:])
        # print()
        
        # Creating Static Test Set
        x_t_aud, y_t_aud = audio_val_batch_generator(self.df_audio_test, batchsize=1)
        x_t_vis, y_t_vis = vis_val_batch_generator(self.df_image_test, batchsize=1)
        y_t = y_t_vis
        # print(x_t_aud.shape)
        # print(x_t_aud[0,:])
        # print(x_t_aud[-1,:])
        # print()

        x_, y_ = audio_batch_generator(self.df_audio, batch_size=1)

        y_pred

        print('Training is starting...')

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            val_iteration = 0
            global_loss = 100
            # Loop over epochs
            for e in range(epochs):
                # Loop over batches
                for (x_aud,y_aud), (x_vis, y_vis) in zip(audio_batch_generator(self.df_audio, batch_size=1), vis_batch_generator(self.df_image, batch_size)):
                    assert np.mean(y_aud) == np.mean(y_vis)
                    y = y_vis
                    
                    # Feed dictionary
                    feed = {inputs_image : x_vis, inputs_audio : x_aud, 
                            labels_ : y, keep_prob_ : 0.3, learning_rate_ : learning_rate}
                    
                    # Loss
                    loss, _ , acc, acc2 = sess.run([cost, optimizer, accuracy, accuracy2], feed_dict = feed)
                    train_acc.append(acc)
                    train_acc2.append(acc2)
                    train_loss.append(loss)
                    
                    # Print at each 5 iters
                    if (iteration % 40 == 0):
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))
                        
                    # Compute validation loss at every 20 iterations
                    if (iteration%20 == 0):                
                        val_acc_ = []
                        val_acc2_ =[]
                        val_loss_ = []

                        tst_acc_ = []
                        tst_acc2_ =[]
                        tst_loss_ = []
                        
                        val_batch_gt = np.empty((0,13), float)
                        val_batch_score = np.empty((0,13), float)

                        tst_batch_gt = np.empty((0,13), float)
                        tst_batch_score = np.empty((0,13), float)
                        
                        # x_v_aud, y_v_aud = audio_val_batch_generator(self.df_audio_test, batchsize=1)
                        # x_v_vis, y_v_vis = vis_val_batch_generator(self.df_image_test, batchsize=1)
                        # y_v = y_v_vis

                        # Feed
                        feed = {inputs_image: x_v_vis, inputs_audio : x_v_aud,  
                                labels_ : y_v, keep_prob_ : 1.0}  

                        # Loss
                        loss_v, acc_v, y_gt_v, y_score_v, acc_2 = sess.run([cost, accuracy, gt_, pr_, accuracy2], feed_dict = feed)           
                        val_acc_.append(acc_v)
                        val_acc2_.append(acc_2)
                        val_loss_.append(loss_v)
                        
                        val_batch_gt = np.append(val_batch_gt, y_gt_v, axis=0)
                        val_batch_score = np.append(val_batch_score, y_score_v, axis=0)
                        
                        if (iteration%40 == 0):       
                            # Print info
                            print("Epoch: {}/{}".format(e, epochs),
                                  "Iteration: {:d}".format(iteration),
                                  "Validation loss: {:6f}".format(np.mean(val_loss_)),
                                  "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                        if iteration>=40:
                            # print("Validation loss",val_loss_)
                            # print("Validation loss len",len(validation_loss))
                            if iteration<= 40:
                                global_loss = validation_loss[val_iteration-1]
                            if global_loss > np.mean(val_loss_):
                                saver.save(sess, save_path=save_dir, global_step=iteration)
                                mes = "This iteration receive better loss: {:.2f} < {:.2f}. Saving session..."
                                print(mes.format(np.mean(val_loss_), global_loss))
                                global_loss = np.mean(val_loss_)
                        # Store
                        validation_acc.append(np.mean(val_acc_))
                        validation_acc2.append(np.mean(val_acc2_))
                        validation_loss.append(np.mean(val_loss_))
                        
                        validation_gt.append(val_batch_gt)
                        validation_pr.append(val_batch_score)

                        val_iteration += 1

                        # x_t_aud, y_t_aud = audio_val_batch_generator(self.df_audio_test, batchsize=1, validation=False)
                        # x_t_vis, y_t_vis = vis_val_batch_generator(self.df_image_test, batchsize=1, validation=False)
                        # y_t = y_t_vis

                        # Feed
                        feed = {inputs_image: x_t_vis, inputs_audio : x_t_aud,  
                                labels_ : y_t, keep_prob_ : 1.0}  

                        # Loss
                        loss_t, acc_t, y_gt_t, y_score_t, acc_2_t = sess.run([cost, accuracy, gt_, pr_, accuracy2], feed_dict = feed)           
                        tst_acc_.append(acc_t)
                        tst_acc2_.append(acc_2_t)
                        tst_loss_.append(loss_t)
                        
                        tst_batch_gt = np.append(tst_batch_gt, y_gt_t, axis=0)
                        tst_batch_score = np.append(tst_batch_score, y_score_t, axis=0)
                        
                        # Store
                        test_acc.append(np.mean(tst_acc_))
                        test_acc2.append(np.mean(tst_acc2_))
                        test_loss.append(np.mean(tst_loss_))
                        
                        test_gt.append(tst_batch_gt)
                        test_pr.append(tst_batch_score)
                    
                    # Iterate 
                    iteration += 1


        return validation_gt, validation_pr, validation_acc, validation_acc2, validation_loss, train_acc, train_loss, train_acc2, test_gt, test_pr, test_acc, test_acc2, test_loss


    def get_trailer_features(self, load=False, dataset=1, trailer_directory='data/', 
                           sequence_dir='data/', audio_directory='/Volumes/TOSHIBA EXT/audio_samples10M', no_audio=False, year_start=2000):
        if dataset==1:
            directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-1m/ratings.dat'
            all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
            pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/pickles/'
            data = dc.get_movielens_1M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)
        elif dataset==10:
            directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-10m/ratings.dat'
            all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
            pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data10/pickles/'
            directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-10m/ratings.dat'
            data = dc.get_movielens_10M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)
            print(data['training'].shape)
        elif dataset==20:
            raise ValueError('20M dataset is not yet implemented')
        else:
            raise ValueError('dataset selection is not valid! Use 1, 10 or 20 as int type')

        # Get Representations of Trailer Frames
        print('Visual Representations are extracting...')
        self.video_processor = AudioVisualEncoder()
        sequences = self.video_processor.extract_visual_features(_dir_=trailer_directory,load=load, seq_dir=sequence_dir)
        self.visual_features = sequences
        print('Done.')

        if not no_audio:
            # Get Representations of Trailer Audio
            print('Audio Representations are extracting...')
            sequences_audio = self.video_processor.extract_audio_features(_dir_=audio_directory)
            self.audio_features = sequences_audio
            print('Done.')

        # data['Titles']['MovieGenre'] = data['Titles'].MovieGenre.apply(lambda x: x.split('|'))
        data['training'] = data['training'][data['training'].Timestamp>=pd.Timestamp(year=year_start, month=1, day=1, hour=1)]
        data['test'] = data['test'][data['test'].Timestamp>=pd.Timestamp(year=year_start, month=1, day=1, hour=1)]
        self.data = data

    def preprocess_dataset(self, rand_state=123, sequence_lenght=60):

        unique_movies_train = self.data['training'].Movie.unique().tolist()
        mov_to_rm = []
        for id_ in self.data['training'].Movie.unique().tolist() + self.data['test'].Movie.unique().tolist():
            if not id_ in list(self.visual_features.keys()):
                # print(id_)
                continue
            if self.visual_features[id_].shape[0] < sequence_lenght:
                mov_to_rm.append(id_)
        unique_movies_train = list(set(unique_movies_train) - set(mov_to_rm))

        unique_movies_test = self.data['test'].Movie.unique().tolist()
        mov_to_rm = []
        for id_ in self.data['training'].Movie.unique().tolist()+self.data['test'].Movie.unique().tolist():
            if not id_ in list(self.visual_features.keys()):
                # print(id_)
                continue
            if self.visual_features[id_].shape[0] < sequence_lenght:
                mov_to_rm.append(id_)
        unique_movies_test = list(set(unique_movies_test) - set(mov_to_rm))

        unique_movies = unique_movies_train #+ unique_movies_test
        # len(unique_movies)
        users_with_features = list(self.data['training'].User.unique())
        users_in_test = list(self.data['test'].User.unique())
        users_without_features = list(set(users_in_test) - set(users_with_features))

        unique_movies_test = self.data['test'][~self.data['test'].User.isin(users_without_features)].Movie.unique().tolist()
        mov_to_rm = []
        for id_ in self.data['training'].Movie.unique().tolist()+self.data['test'][~self.data['test'].User.isin(users_without_features)].Movie.unique().tolist():
            if not id_ in list(self.visual_features.keys()):
                # print(id_)
                continue
            if self.visual_features[id_].shape[0] < sequence_lenght:
                mov_to_rm.append(id_)
        unique_movies_test = list(set(unique_movies_test) - set(mov_to_rm))
        unique_movies = unique_movies_train #+ unique_movies_test

        random.seed(rand_state)
        random_unique_movies = random.sample(unique_movies,250)
        unique_movies_train = list(set(unique_movies) - set(random_unique_movies))
        unique_movies_test = unique_movies_test # random_unique_movies[:250]
        unique_movies_validation = random_unique_movies #[250:]
        print('Training Sample:',len(unique_movies_train))
        print('Validation Sample:',len(unique_movies_validation))
        print('Test Sample:',len(unique_movies_test))

        df_audio = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_train, total=len(unique_movies_train), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), 60)),:]) 
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    pass
                    # print(sample)
        df_audio = pd.concat(df_list, axis=0)
        del df_list
        # print(df_audio.shape)

        df_audio_val = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_validation, total=len(unique_movies_validation), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), 60)),:]) 
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    pass
                    # print(sample)
        df_audio_val = pd.concat(df_list, axis=0)
        del df_list
        # print(df_audio_val.shape)

        df_audio_test = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_test, total=len(unique_movies_test), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), 60)),:]) 
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    pass
                    # print(sample)
        df_audio_test = pd.concat(df_list, axis=0)
        del df_list
        # print(df_audio_test.shape)

        df_image = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_train, total=len(unique_movies_train), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), 60)),:]) 
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    print(sample)
        df_image = pd.concat(df_list, axis=0)
        del df_list
        # print(df_image.shape)

        df_image_val = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_validation, total=len(unique_movies_validation), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), 60)),:]) 
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    print(sample)
        df_image_val = pd.concat(df_list, axis=0)
        del df_list
        # print(df_image_val.shape)

        df_image_test = pd.DataFrame()
        df_list = []
        for sample in tqdm(unique_movies_test, total=len(unique_movies_test), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = pd.DataFrame(self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), 60)),:]) 
                    df_tmp['movie_id'] = sample
                    df_list.append(df_tmp)
                else:
                    print(sample)
        df_image_test = pd.concat(df_list, axis=0)
        del df_list
        # print(df_image_test.shape)

        df_image = df_image[df_image.movie_id.isin(df_audio.movie_id.unique().tolist())]

        train_unique_movies_with_features = df_image.movie_id.unique().tolist()
        # print(len(train_unique_movies_with_features))

        val_unique_movies_with_features = df_image_val.movie_id.unique().tolist()
        # print(len(val_unique_movies_with_features))

        test_unique_movies_with_features = df_image_test.movie_id.unique().tolist()
        # print(len(test_unique_movies_with_features))

        
        #############################
        # Get Latent Factors Training
        #############################

        data = pd.concat([self.data['training'],self.data['test']],axis=0).sample(frac=1, random_state=1321, axis=0).reset_index(drop=True)
        data_train = data[~data.Movie.isin(val_unique_movies_with_features+test_unique_movies_with_features)]
        data_val = data[data.Movie.isin(val_unique_movies_with_features)]
        data_test = data[data.Movie.isin(test_unique_movies_with_features)]
        print(data_train.shape)
        print(data_val.shape)
        print(data_test.shape)


        print('Training User-Movie Latent Factors are extracting...')
        print(data_train.shape)
        self.user_item_network_training = CollaborativeFiltering(data_train) # 0 Threshold trim
        user_latent_traninig, movie_latent_traninig, sigma = self.user_item_network_training.compute_latent_factors(algorithm='SVD', k=100)
        self.user_latent_traninig = user_latent_traninig
        self.movie_latent_traninig = movie_latent_traninig
        print('Done.')

        # self.df_image = df_image
        # self.df_image_val = df_image_val
        # self.df_image_test = df_image_test
        # self.df_audio = df_audio
        # self.df_audio_val = df_audio_val
        # self.df_audio_test = df_audio_test

        df_user = pd.DataFrame()
        df_list = []
        for sample in tqdm(data_train.User.unique().tolist(), total=data_train.User.nunique(), position=0):
            if sample in list(self.user_latent_traninig.keys()):
                df_tmp = pd.DataFrame(self.user_latent_traninig[sample]).T
                df_list.append(df_tmp)
        df_user = pd.concat(df_list,axis=0)
        del df_list
        print('User dataframe shape:',df_user.shape)

        self.df_user = df_user

        ratings_df_training_filtered = data_train[data_train.Rating.isin([0.5,1.,1.5,2.0,2.5,4.5,5.])].copy()#.sample(frac=1, random_state=1321)
        ratings_df_val_filtered = data_val[data_val.Rating.isin([0.5,1.,1.5,2.0,2.5,4.5,5.])].copy()
        ratings_df_test_filtered = data_test[data_test.Rating.isin([0.5,1.,1.5,2.0,2.5,4.5,5.])].copy()#.sample(frac=1, random_state=1321)
        
        print('Trailer dictinaries are being created\n')

        dict_audio = {}
        for sample in tqdm(train_unique_movies_with_features , total=len(train_unique_movies_with_features), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), 60)),:]
                    dict_audio[sample] = df_tmp
                else:
                    print(sample)
        # print(len(dict_audio.keys()))

        dict_audio_val = {}
        for sample in tqdm(val_unique_movies_with_features, total=len(val_unique_movies_with_features), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), 60)),:] 
                    dict_audio_val[sample] = df_tmp
                else:
                    print(sample)
        # print(len(dict_audio_val.keys()))

        dict_audio_test = {}
        for sample in tqdm(test_unique_movies_with_features, total=len(test_unique_movies_with_features), position=0):
            if sample in list(self.audio_features.keys()):
                if self.audio_features[sample].shape[0] >= 60:
                    df_tmp = self.audio_features[sample][np.sort(random.sample(range(self.audio_features[sample].shape[0]), 60)),:] 
                    dict_audio_test[sample] = df_tmp
                else:
                    print(sample)
        # print(len(dict_audio_test.keys()))


        dict_image = {}
        for sample in tqdm(train_unique_movies_with_features, total=len(train_unique_movies_with_features), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), 60)),:]
                    dict_image[sample] = df_tmp
                else:
                    print(sample)
        # print(len(dict_image.keys()))

        dict_image_val = {}
        for sample in tqdm(val_unique_movies_with_features, total=len(val_unique_movies_with_features), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), 60)),:]
                    dict_image_val[sample] = df_tmp
                else:
                    print(sample)
        # print(len(dict_image_val.keys()))

        dict_image_test = {}
        for sample in tqdm(test_unique_movies_with_features, total=len(test_unique_movies_with_features), position=0):
            if sample in list(self.visual_features.keys()):
                if self.visual_features[sample].shape[0] >= 60:
                    df_tmp = self.visual_features[sample][np.sort(random.sample(range(self.visual_features[sample].shape[0]), 60)),:]
                    dict_image_test[sample] = df_tmp
                else:
                    print(sample)
        # print(len(dict_image_test.keys()))
        # print('Movies omited completed')

        self.df_image = dict_image
        self.df_image_val = dict_image_val
        self.df_image_test = dict_image_test
        self.df_audio = dict_audio
        self.df_audio_val = dict_audio_val
        self.df_audio_test = dict_audio_test

        self.processed_data = (data_train, data_val, data_test)

        return ratings_df_training_filtered, ratings_df_val_filtered, ratings_df_test_filtered

    # def train(self, system):
    #     if system == 'Basic':
    #         model = Model1(self.user_item_network_training.CF_data, self.user_latent_traninig, self.movie_factors_training,
    #                        self.user_item_network_test.CF_data, self.user_latent_test, self.movie_factors_test, self.visual_features, self.output)
    #     elif system == 'SVM':
    #         SVM_model(self.user_factors, self.movie_factors, self.trailer_sequence_representation, self.output)

##################################################

# AUDIO+IMAGE+CONCAT_withoutnormalization
# cv1 = {'Mean Accuracy': 0.88258743,
#  'Std Accuracy': 0.0016023071,
#  'All Label True Mean Accuracy': 0.22681819,
#  'All Label True Std Accuracy': 0.016879492,
#  'Mean ROC Curve Area': 0.862391437501645,
#  'Std ROC Curve Area': 0.0034494904499349725}

# roc_mean = {'Thriller': 0.8156934893199929,
#  'War': 0.8633896151080437,
#  'Adventure': 0.8173073421040541,
#  'Comedy': 0.8276088691873474,
#  'Children': 0.8720645717509606,
#  'Crime': 0.7795350619099958,
#  'Mystery': 0.6903384404313739,
#  'Romance': 0.7109978548460507,
#  'Action': 0.8146620601146088,
#  'Drama': 0.7466068367320481,
#  'Sci-Fi': 0.8161211442633969,
#  'Horror': 0.8593519730960097,
#  'Fantasy': 0.780166976429598}

# roc_std = {'Thriller': 0.01604867402131482,
#  'War': 0.0430477570656686,
#  'Adventure': 0.016611087103345735,
#  'Comedy': 0.009474727435399071,
#  'Children': 0.02394516441156878,
#  'Crime': 0.03309559096842231,
#  'Mystery': 0.03133838914817679,
#  'Romance': 0.034333799149719214,
#  'Action': 0.010253728203237502,
#  'Drama': 0.023189904250268237,
#  'Sci-Fi': 0.02360037819732559,
#  'Horror': 0.018400924247650094,
#  'Fantasy': 0.018587423282192334}

##################################################

# AUDIO+IMAGE+CONCAT_withnormalization
# cv2 = {'Mean Accuracy': 0.8566434,
#  'Std Accuracy': 0.0034894967,
#  'All Label True Mean Accuracy': 0.096363634,
#  'All Label True Std Accuracy': 0.011991735,
#  'Mean ROC Curve Area': 0.7848487392384401,
#  'Std ROC Curve Area': 0.004866283456038735}


# roc_mean = {'War': 0.5951429985298528,
#  'Fantasy': 0.6787317400015357,
#  'Romance': 0.4742464458769121,
#  'Mystery': 0.5563404146014822,
#  'Horror': 0.693803708787126,
#  'Children': 0.6341641571044534,
#  'Action': 0.7391052577281614,
#  'Crime': 0.5906080414466668,
#  'Sci-Fi': 0.735991042045341,
#  'Adventure': 0.6920414441395178,
#  'Drama': 0.5799510287009394,
#  'Comedy': 0.6744497362103676,
#  'Thriller': 0.6659738287682632}

#  roc_std = {'War': 0.028810147523957715,
#  'Fantasy': 0.05464351945389523,
#  'Romance': 0.07534565126876623,
#  'Mystery': 0.03738068429046518,
#  'Horror': 0.03455952389244684,
#  'Children': 0.071791594659351,
#  'Action': 0.012607604592011512,
#  'Crime': 0.023704530532506055,
#  'Sci-Fi': 0.0426345485927024,
#  'Adventure': 0.03638958080482277,
#  'Drama': 0.018493468604396816,
#  'Comedy': 0.02717203946809617,
#  'Thriller': 0.01999308823126566}

##################################################

# AUDIO_withoutnormalization
# cv3 = {'Mean Accuracy': 0.8567832,
#  'Std Accuracy': 0.0033023034,
#  'All Label True Mean Accuracy': 0.08227273,
#  'All Label True Std Accuracy': 0.013071641,
#  'Mean ROC Curve Area': 0.7609355525403347,
#  'Std ROC Curve Area': 0.009278733417821662}

# roc_mean = {'Action': 0.6694844396188158, 'Fantasy': 0.5775772486529052, 'Romance': 0.5227811804757734, 'Horror': 0.6240906636018682, 'Thriller': 0.6541275608905741, 'Comedy': 0.5382046286691778, 'Children': 0.5228208500692837, 'Mystery': 0.5177230123788217, 'Drama': 0.540891681285367, 'War': 0.4623496742072508, 'Adventure': 0.5846145449709206, 'Crime': 0.5815347800897243, 'Sci-Fi': 0.5735259438019917}
# roc_std = {'Action': 0.028077001970071715, 'Fantasy': 0.02366575370394965, 'Romance': 0.06232828383984626, 'Horror': 0.024052757248398487, 'Thriller': 0.021193247205747646, 'Comedy': 0.014297660203609517, 'Children': 0.009776906994166093, 'Mystery': 0.07111250002266244, 'Drama': 0.013729560496941988, 'War': 0.04274177042156962, 'Adventure': 0.05372238549279903, 'Crime': 0.015412961878973272, 'Sci-Fi': 0.03914911997929026}

##################################################

# IMAGE_withoutnormalization
# cv4 = {'Mean Accuracy': 0.88076925,
 # 'Std Accuracy': 0.0020086048,
 # 'All Label True Mean Accuracy': 0.22772726,
 # 'All Label True Std Accuracy': 0.0067266584,
 # 'Mean ROC Curve Area': 0.863467688070319,
 # 'Std ROC Curve Area': 0.00445024455445815} 

# roc_mean = {'Action': 0.8186787694120361, 'Fantasy': 0.7812694456650925, 'Romance': 0.7152244390195999, 'Horror': 0.8710621326606052, 'Thriller': 0.8157369507111327, 'Comedy': 0.8227397965386529, 'Children': 0.8709111170846265, 'Mystery': 0.6897914549958738, 'Drama': 0.7453202101013374, 'War': 0.8731432037001629, 'Adventure': 0.822235750132432, 'Crime': 0.7831921200099816, 'Sci-Fi': 0.8172443102732905}
# roc_std = {'Action': 0.013628864969374488, 'Fantasy': 0.01725093295806066, 'Romance': 0.02898298004485256, 'Horror': 0.01594754569510265, 'Thriller': 0.01813498320203215, 'Comedy': 0.009169188635807264, 'Children': 0.03238073792084603, 'Mystery': 0.06501560396884323, 'Drama': 0.01809750534602589, 'War': 0.027527814137953497, 'Adventure': 0.006820565624929191, 'Crime': 0.017863903921767423, 'Sci-Fi': 0.013899393439803683}

##################################################

# AUDIO_withoutnormalization LONGER CONV
# cv5 = {'Mean Accuracy': 0.8615385,
 # 'Std Accuracy': 0.0035835353,
 # 'All Label True Mean Accuracy': 0.15318182,
 # 'All Label True Std Accuracy': 0.007201352,
 # 'Mean ROC Curve Area': 0.7743230292087707,
 # 'Std ROC Curve Area': 0.00953435853380593}

# roc_mean = {'Drama': 0.6129476460780375, 'Horror': 0.6784597076792401, 'Mystery': 0.5639306785509051, 'Children': 0.6521179088147059, 'Fantasy': 0.5898192253433255, 'Comedy': 0.6636830328495718, 'Action': 0.7180541774064807, 'Thriller': 0.6828717631656498, 'Crime': 0.5976217114954131, 'Sci-Fi': 0.590155819375545, 'Adventure': 0.6054525861319306, 'Romance': 0.616806739567232, 'War': 0.559683351508582}
# roc_std = {'Drama': 0.01301236608540371, 'Horror': 0.025506827625963065, 'Mystery': 0.053087970142405655, 'Children': 0.006250805404489619, 'Fantasy': 0.03177819307168222, 'Comedy': 0.009064936015943286, 'Action': 0.016495462657135796, 'Thriller': 0.009675954620205698, 'Crime': 0.0016221588732230024, 'Sci-Fi': 0.0136313151627276, 'Adventure': 0.03781460632649885, 'Romance': 0.02297890752361728, 'War': 0.036800741977944904}

##################################################

# AUDIO+IMAGE_withoutnormalization LONGER CONV
# cv6 = {'Mean Accuracy': 0.88160837,
 # 'Std Accuracy': 0.0022257015,
 # 'All Label True Mean Accuracy': 0.22409092,
 # 'All Label True Std Accuracy': 0.015870083,
 # 'Mean ROC Curve Area': 0.8624268278297264,
 # 'Std ROC Curve Area': 0.004983447945534734}

# roc_mean = {'Drama': 0.7456476796368346, 'Horror': 0.8774635549287043, 'Mystery': 0.6918174062761011, 'Children': 0.8590721837246669, 'Fantasy': 0.778587563066819, 'Comedy': 0.8310348068255777, 'Action': 0.8251764728296171, 'Thriller': 0.8142653907251465, 'Crime': 0.7715858396469976, 'Sci-Fi': 0.8155725698727301, 'Adventure': 0.8133753780442903, 'Romance': 0.7215352675193157, 'War': 0.8503230254880946}
# roc_std = {'Drama': 0.024634634393965862, 'Horror': 0.003932585807469424, 'Mystery': 0.057189601936124206, 'Children': 0.03600999678900703, 'Fantasy': 0.009094792921466675, 'Comedy': 0.011172122726914775, 'Action': 0.006282805856931287, 'Thriller': 0.0057506587365138435, 'Crime': 0.020817788853193894, 'Sci-Fi': 0.02295664518572803, 'Adventure': 0.012141137045283289, 'Romance': 0.02197698316702992, 'War': 0.050432714802689335}


