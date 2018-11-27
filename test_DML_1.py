import time
import datetime
import gc
from random import random

from collections import Counter
from itertools import combinations
import tqdm

from multimodalrec.multimodalrec import MultimodalRec
from multimodalrec.multimodalrec import data_pipeline
from multimodalrec.model import model

from multimodalrec.model import SiameseLSTM
from data import data_creation as dc

import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import datetime
import gc
from random import random
from random import shuffle

# Parameters
emdding_dim = 2048
dropout_keep_prob = 1.
l2_reg_lambda = 0.
hidden_units = 50
frame_size = 30

# Training Parameters
batch_size = 64
num_epochs = 300
evaluate_every = 500
checkpoint_every = 1000
learning_rate = 1e-3


def test_generator(test_similarity_df, recmodel):
    x1_batch, x2_batch, y_batch = [], [], []
    for ind, row in test_similarity_df.iterrows():
        x1_batch.append(recmodel.visual_features[row['Movie1']])
        x2_batch.append(recmodel.visual_features[row['Movie2']])
        y_batch.append(row['Similarity'])

    return (np.swapaxes(np.swapaxes(np.dstack(x1_batch),0,2),1,2), np.swapaxes(np.swapaxes(np.dstack(x2_batch),0,2),1,2), np.array(y_batch))


def create_validation_set(data, recmodel):
    # Test Pos
    positive_labels_test = []
    for user in tqdm.tqdm(data['test'].User.unique().tolist(), total=len(data['test'].User.unique().tolist()), position=0):
        user_df = data['test'][(data['test'].User == user)&(data['test'].Rating > 4)]
        if user_df.shape[0] < 2: continue
        user_movies = user_df.Movie.tolist()
        C = list(combinations(user_movies, 2))
        C = [(c[0], c[1], 1) for c in C]
        positive_labels_test += C
    print(len(positive_labels_test))
    cntr_pos_test = Counter(positive_labels_test)
    print(max(list(cntr_pos_test.values())))
    print(np.mean(list(cntr_pos_test.values())))
    cntr_pos_test = dict(cntr_pos_test)
    unique_pos_labels_test = []
    for key in cntr_pos_test.keys(): 
        if cntr_pos_test[key] > 5:
            unique_pos_labels_test.append(key)
    len(unique_pos_labels_test)
    # Test Neg
    negative_labels_test = []
    for user in tqdm.tqdm(data['test'].User.unique().tolist(), total=len(data['test'].User.unique().tolist()), position=0):
        user_df = data['test'][(data['test'].User == user)&(data['test'].Rating < 3)]
        if user_df.shape[0] < 2: continue
        user_movies = user_df.Movie.tolist()
        C = list(combinations(user_movies, 2))
        C = [(c[0], c[1], 0) for c in C]
        negative_labels_test += C
    print(len(negative_labels_test))
    cntr_neg_test = Counter(negative_labels_test)
    print(max(list(cntr_neg_test.values())))
    print(np.mean(list(cntr_neg_test.values())))
    cntr_neg_test = dict(cntr_neg_test)
    unique_neg_labels_test = []
    for key in cntr_neg_test.keys(): 
        if cntr_neg_test[key] > 10:
            unique_neg_labels_test.append(key)
    len(unique_neg_labels_test)

    test_pairs = unique_pos_labels_test+unique_neg_labels_test
    shuffle(test_pairs)

    test_similarity_df = pd.DataFrame(np.array(test_pairs), columns=['Movie1','Movie2','Similarity'])

    mov_to_rm = []
    for id_ in data['test'].Movie.unique().tolist():
        if recmodel.visual_features[id_].shape != (30, 2048):
            mov_to_rm.append(id_)

    test_similarity_df = test_similarity_df[~test_similarity_df.Movie1.isin(mov_to_rm)]
    test_similarity_df = test_similarity_df[~test_similarity_df.Movie2.isin(mov_to_rm)]
    
    return test_similarity_df


def batch_generator(similarity_df,recmodel, batchsize=64):
    for batch in range(1,int(similarity_df.shape[0]/64)):
        batch_df = similarity_df.iloc[(batch-1)*64:(batch)*64]
        x1_batch, x2_batch, y_batch = [], [], []
        for ind, row in batch_df.iterrows():
            x1_batch.append(recmodel.visual_features[row['Movie1']])
            x2_batch.append(recmodel.visual_features[row['Movie2']])
            y_batch.append(row['Similarity'])
        
        yield (np.swapaxes(np.swapaxes(np.dstack(x1_batch),0,2),1,2), np.swapaxes(np.swapaxes(np.dstack(x2_batch),0,2),1,2), np.array(y_batch))


def main():
    recmodel = MultimodalRec()
    recmodel.organize_multimodal_data(load=True)
    ratings_df_training = recmodel.user_item_network_training.CF_data

    directory = os.path.dirname(os.path.realpath("__file__"))+'/data/ml-1m/ratings.dat'
    all_movies_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/all_movies.txt'
    pickles_dir = os.path.dirname(os.path.realpath("__file__"))+'/data/pickles/'

    data = dc.get_movielens_1M(directory=directory, all_movies_dir=all_movies_dir, pickles_dir=pickles_dir, min_positive_score=0)
    
    positive_labels = []
    for user in tqdm.tqdm(data['training'].User.unique().tolist(), total=len(data['training'].User.unique().tolist()), position=0):
        user_df = data['training'][(data['training'].User == user)&(data['training'].Rating > 4)]
        if user_df.shape[0] < 2: continue
        user_movies = user_df.Movie.tolist()
        C = list(combinations(user_movies, 2))
        C = [(c[0], c[1], 1) for c in C]
        positive_labels += C
    len(positive_labels)

    cntr_pos = Counter(positive_labels)
    print(max(list(cntr_pos.values())))
    print(np.mean(list(cntr_pos.values())))

    cntr_pos = dict(cntr_pos)
    unique_pos_labels = []
    for key in cntr_pos.keys(): 
        if cntr_pos[key] > 5:
            unique_pos_labels.append(key)
    len(unique_pos_labels)

    negative_labels = []
    for user in tqdm.tqdm(data['training'].User.unique().tolist(), total=len(data['training'].User.unique().tolist()), position=0):
        user_df = data['training'][(data['training'].User == user)&(data['training'].Rating < 3)]
        if user_df.shape[0] < 2: continue

        user_movies = user_df.Movie.tolist()
        C = list(combinations(user_movies, 2))
        C = [(c[0],c[1],0) for c in C]
        negative_labels += C
    len(negative_labels)

    cntr_neg = Counter(negative_labels)
    print(max(list(cntr_neg.values())))
    print(np.mean(list(cntr_neg.values())))

    cntr_neg = dict(cntr_neg)
    unique_neg_labels = []
    for key in cntr_neg.keys(): 
        if cntr_neg[key] > 3:
            unique_neg_labels.append(key)
    len(unique_neg_labels)

    training_pairs = unique_pos_labels+unique_neg_labels

    shuffle(training_pairs)

    similarity_df = pd.DataFrame(np.array(training_pairs), columns=['Movie1','Movie2','Similarity'])

    mov_to_rm = []
    for id_ in data['training'].Movie.unique().tolist():
        if recmodel.visual_features[id_].shape != (30, 2048):
            mov_to_rm.append(id_)

    similarity_df = similarity_df[~similarity_df.Movie1.isin(mov_to_rm)]
    similarity_df = similarity_df[~similarity_df.Movie2.isin(mov_to_rm)]

    test_similarity_df = create_validation_set(data, recmodel)

    tf.reset_default_graph()

    print("starting graph def")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto()
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            siameseModel = SiameseLSTM(frame_size=frame_size, 
                                       embedding_size=emdding_dim, 
                                       hidden_units=hidden_units,
                                       batch_size=batch_size, 
                                       l2_reg_lambda=l2_reg_lambda)
            global_step = tf.Variable(0, name = "global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            print("Initialized siamese object")

        grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
        tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        print("defined training_ops")

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        print("defined gradient summaries")

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs1"))#, timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", siameseModel.loss)
        acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        #def train_step(x1_batch,x2_batch,y_batch):
        #    """A Single tranining step"""

        print("init all variables")
        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
            f.write(graphpb_txt)

        # Generate batches
        #batch_generator

        ptr=0
        max_validation_acc=0.0
        #for nn in range(noBatches*num_epochs):
        for nn, next_batch in enumerate(batch_generator(similarity_df, batchsize = batch_size, recmodel=recmodel)):
            #batch = batches.next()
            #if len(batch)<1:
            #    continue
            x1_batch, x2_batch, y_batch = next_batch
            if len(y_batch)<1:
                continue

            # TRAINING
            if random()>.5:
                feed_dict = {
                    siameseModel.input_x1: x1_batch,
                    siameseModel.input_x2: x2_batch,
                    siameseModel.input_y: y_batch,
                    siameseModel.dropout_keep_prob: dropout_keep_prob,
                }
            else:
                feed_dict = {
                    siameseModel.input_x1: x2_batch,
                    siameseModel.input_x2: x1_batch,
                    siameseModel.input_y: y_batch,
                    siameseModel.dropout_keep_prob: dropout_keep_prob,
                }

            _, step, loss, accuracy, dist, sim, summaries = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, 
                                                                      siameseModel.distance, siameseModel.temp_sim, train_summary_op],  feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            print(y_batch, dist, sim)
            #train_step(x1_batch,x2_batch,y_batch)

            current_step = tf.train.global_step(sess, global_step)
            sum_acc = 0.

            # VALIDATION SET EVALUATION
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                x1_dev, x2_dev, y_dev = test_generator(test_similarity_df, recmodel)

                if random()>0.5:
                    feed_dict = {
                        siameseModel.input_x1: x1_batch,
                        siameseModel.input_x2: x2_batch,
                        siameseModel.input_y: y_batch,
                        siameseModel.dropout_keep_prob: 1.0,
                    }
                else:
                    feed_dict = {
                        siameseModel.input_x1: x2_batch,
                        siameseModel.input_x2: x1_batch,
                        siameseModel.input_y: y_batch,
                        siameseModel.dropout_keep_prob: 1.0,
                    }

                step, loss, accuracy, sim, summaries = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, 
                                                                siameseModel.temp_sim, dev_summary_op], feed_dict)
                print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                dev_summary_writer.add_summary(summaries, step)
                print("")

            if current_step % checkpoint_every == 0:
                if sum_acc >= max_validation_acc:
                    max_validation_acc = sum_acc
                    saver.save(sess, checkpoint_prefix, global_step=current_step)
                    tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                    print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))


if __name__ == '__main__':
    main()