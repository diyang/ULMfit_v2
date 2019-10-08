import os
import numpy as np
import pandas as pd
import tensorflow as tf
from customLSTMcell import LayerNormalizedLSTMCell
import matplotlib.pyplot as plt
import math
import multilabel_evaluation as m_eval
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from prettytable import PrettyTable

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


class RNN_bidirect_build_graph:
    def __init__(self, state_size,
                 num_words,
                 num_classes,
                 batch_size,
                 sequence_length,
                 cell_type=None,
                 num_layers=3,
                 init_trainable=False): 

        self.state_size = state_size
        self.num_words = num_words
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.init_trainable = init_trainable
        self.graph = self.build_graph()
        self.sess = None

    def get_graph(self):
        return self.graph

    def build_graph(self):
        reset_graph()
        with tf.variable_scope("rnn_layer"):
            x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name='input_placeholder')
            y_word = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name='labels_placeholder')
            y_sentiment = tf.placeholder(tf.float32, [self.batch_size, self.num_classes],
                                         name='sentiments_placeholder')  ## tf.int32
            label_weight = tf.placeholder(tf.float32, [self.num_classes, self.num_classes], 
                                         name='label_weight_placeholder')
            embeddings = tf.placeholder(tf.float32, [self.num_words, self.state_size], name='embeddings_placeholder')

            tf_dropout = tf.placeholder_with_default(1.0, [])
            tf_learning_rate = tf.placeholder_with_default(1e-4, [])

            rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

            if self.cell_type == 'GRU':
                cell_fw = tf.nn.rnn_cell.GRUCell(self.state_size)
                cell_bw = tf.nn.rnn_cell.GRUCell(self.state_size)
            elif self.cell_type == 'LSTM':
                cell_fw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
                cell_bw = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
            elif self.cell_type == 'LN_LSTM':
                cell_fw = LayerNormalizedLSTMCell(self.state_size)
                cell_bw = LayerNormalizedLSTMCell(self.state_size)
            else:
                cell_fw = tf.nn.rnn_cell.BasicRNNCell(self.state_size)
                cell_bw = tf.nn.rnn_cell.BasicRNNCell(self.state_size)

            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=tf_dropout)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=tf_dropout)

            if self.cell_type == 'LSTM' or self.cell_type == 'LN_LSTM':
                cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)
            else:
                cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.num_layers)
                cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.num_layers)

            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=tf_dropout)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=tf_dropout)

            if self.init_trainable:
                init_state_fw = []
                init_state_bw = []
                if self.cell_type == 'LSTM' or self.cell_type == 'LN_LSTM':
                    for rnn_layer in range(self.num_layers):
                        init_fw_c = tf.dtypes.cast(
                            tf.get_variable('init_' + str(rnn_layer) + '_fw_c', [self.batch_size, self.state_size],
                                            trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_fw_h = tf.dtypes.cast(
                            tf.get_variable('init_' + str(rnn_layer) + '_fw_h', [self.batch_size, self.state_size],
                                            trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_bw_c = tf.dtypes.cast(
                            tf.get_variable('init_' + str(rnn_layer) + '_bw_c', [self.batch_size, self.state_size],
                                            trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_bw_h = tf.dtypes.cast(
                            tf.get_variable('init_' + str(rnn_layer) + '_bw_h', [self.batch_size, self.state_size],
                                            trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_state_fw.append(tf.nn.rnn_cell.LSTMStateTuple(init_fw_c, init_fw_h))
                        init_state_bw.append(tf.nn.rnn_cell.LSTMStateTuple(init_bw_c, init_bw_h))
                else:
                    for rnn_layer in range(self.num_layers):
                        init_fw_s = tf.dtypes.cast(
                            tf.get_variable('init_' + str(rnn_layer) + '_fw_s', [self.batch_size, self.state_size],
                                            trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_bw_s = tf.dtypes.cast(
                            tf.get_variable('init_' + str(rnn_layer) + '_bw_s', [self.batch_size, self.state_size],
                                            trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_state_fw.append(init_fw_s)
                        init_state_bw.append(init_bw_s)
                init_state_fw = tuple(init_state_fw)
                init_state_bw = tuple(init_state_bw)
            else:
                init_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
                init_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

            (rnn_outputs_fw, rnn_outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=rnn_inputs,
                initial_state_fw=init_state_fw,
                initial_state_bw=init_state_bw)
            rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_bw], 2)
            rnn_outputs = tf.identity(rnn_outputs, 'rnn_outputs_tensor')

        # -------------------------------- NEXT WORDING PREDICTION --------------------------------------------
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate_schedule = tf.train.cosine_decay(self.learning_rate, global_step, 5)
        with tf.variable_scope('word_prediction'):
            # W1 = tf.get_variable('W1', [self.state_size*2, self.state_size])
            # b1 = tf.get_variable('b1', [self.state_size], initializer=tf.constant_initializer(0.0))
            W2 = tf.get_variable('W2', [self.state_size * 2, self.num_words])
            b2 = tf.get_variable('b2', [self.num_words], initializer=tf.constant_initializer(0.0))
            rnn_outputs_reshape = tf.reshape(rnn_outputs, [-1, self.state_size * 2])
            # fc_word = tf.matmul(rnn_outputs_reshape, W1) + b1
            word_logits = tf.matmul(rnn_outputs_reshape, W2) + b2
            word_predictions = tf.nn.softmax(word_logits, name='predictions')
            y_reshaped = tf.reshape(y_word, [-1])
            word_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_reshaped, logits=word_logits,
                                                                                name="word_cross_entropy")
            word_cross_entropy_mean = tf.reduce_mean(word_cross_entropy, name="word_cross_entropy_mean")
            word_train_step = tf.train.AdamOptimizer(tf_learning_rate).minimize(word_cross_entropy_mean)

        # ------------------------------- SENTIMENT CLASSIFICATION --------------------------------------------
        with tf.variable_scope('sentiment_softmax'):
            # Hierarchical Attention Layer
            W_omega = tf.get_variable('W_omega', [self.state_size * 2, self.state_size])
            b_omega = tf.get_variable('b_omega', [self.state_size])
            u_omega = tf.get_variable('u_omega', [self.state_size])
            v = tf.tanh(tf.tensordot(rnn_outputs, W_omega, axes=1) + b_omega)
            vu = tf.tensordot(v, u_omega, axes=1)
            alphas = tf.nn.softmax(vu)
            rnn_outputs_attn = tf.reduce_sum(rnn_outputs * tf.expand_dims(alphas, -1), 1)
            # Dense Layers
            rnn_outputs_lst = tf.squeeze(
                tf.slice(rnn_outputs, [0, (self.sequence_length - 1), 0], [self.batch_size, 1, self.state_size * 2]))
            rnn_outputs_concat = tf.concat([rnn_outputs_lst, rnn_outputs_attn], 1)
            W_s1 = tf.get_variable('W_s1', [self.state_size * 4, self.state_size * 2])
            b_s1 = tf.get_variable('b_s1', [self.state_size * 2], initializer=tf.constant_initializer(0.0))
            W_s2 = tf.get_variable('W_s2', [self.state_size * 2, self.state_size])
            b_s2 = tf.get_variable('b_s2', [self.state_size], initializer=tf.constant_initializer(0.0))
            W_s3 = tf.get_variable('W_s3', [self.state_size, self.num_classes])
            b_s3 = tf.get_variable('b_s3', [self.num_classes], initializer=tf.constant_initializer(0.0))
            fc1 = tf.matmul(rnn_outputs_concat, W_s1) + b_s1
            fc2 = tf.matmul(fc1, W_s2) + b_s2
            sentiment_logits = tf.matmul(fc2, W_s3) + b_s3
            sentiment_predictions = tf.nn.sigmoid(sentiment_logits, name='predictions_sentiment')
            sentiment_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_sentiment,
                                                                              logits=sentiment_predictions,
                                                                              name="cross_entropy_sentiment")
            sentiment_cross_entropy_weighted = tf.matmul(sentiment_cross_entropy, label_weight)
            sentiment_cross_entropy_mean = tf.reduce_mean(sentiment_cross_entropy_weighted, 
                                                          name="cross_entropy_sentiment_mean")

        freeze_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sentiment_softmax")
        sentiment_train_step_freeze = tf.train.AdamOptimizer(tf_learning_rate).minimize(sentiment_cross_entropy_mean,
                                                                                        var_list=freeze_train_vars)
        sentiment_train_step_unfreeze = tf.train.AdamOptimizer(tf_learning_rate).minimize(sentiment_cross_entropy_mean)

        return dict(
            x=x,
            y_word=y_word,
            y_sentiment=y_sentiment,
            label_weight = label_weight,
            embeddings=embeddings,
            dropout=tf_dropout,
            learning_rate=tf_learning_rate,

            init_state_fw=init_state_fw,
            init_state_bw=init_state_bw,
            final_state_fw=final_state_fw,
            final_state_bw=final_state_bw,

            total_loss_word=word_cross_entropy_mean,
            train_step_word=word_train_step,
            preds_word=word_predictions,

            total_loss_sentiment=sentiment_cross_entropy_mean,
            train_step_sentiment_freeze=sentiment_train_step_freeze,
            train_step_sentiment_unfreeze=sentiment_train_step_unfreeze,
            preds_sentiment=sentiment_predictions,

            freeze_check=rnn_outputs_reshape,
            saver=tf.train.Saver()
        )

    def train_base_model(self, data, num_epochs, learning_rate, zero_init_state=True, verbose=True, verbose_graph=False,
                         save=False, pretrain_model=None, dropout=None):
        tf.set_random_seed(2345)
        steps_per_train_epoch = int(data['train_data'].shape[0] / self.batch_size) + 1
        if 'test_data' in data.keys():
            steps_per_test_epoch = int(data['test_data'].shape[0] / self.batch_size) + 1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            testing_losses = []
            if pretrain_model is not None:
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_model))
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading Model ...")
                    self.graph["saver"].restore(sess, ckpt.model_checkpoint_path)
                    print("Pre-trained Model Loaded")
                    if os.path.exists("bidrect_base_training_losses.npy"):
                        training_losses = np.load("bidrect_base_training_losses.npy").tolist()
                    if os.path.exists("bidrect_base_testing_losses.npy"):
                        testing_losses = np.load("bidrect_base_testing_losses.npy").tolist()

            print("training start .....")

            if verbose_graph:
                fig, (ax_loss, ax_loss2) = plt.subplots(2, 1, figsize=(8, 8))
                ax_loss.set_ylabel("Cross Entropy")
                ax_loss.set_xlabel("Epoches")
                ax_loss2.set_ylabel("Cross Entropy")
                ax_loss2.set_xlabel("Epoches")
                plt.ion()
                fig.show()
                fig.canvas.draw()

            for epoch in range(num_epochs):
                print("in epoch ", epoch)  ##debug
                training_loss = 0
                training_state_fw = None
                training_state_bw = None
                for step in range(steps_per_train_epoch):
                    offset = (step * self.batch_size) % (data['train_data'].shape[0] - self.batch_size)
                    batch_data = data['train_data'][offset:(offset + self.batch_size)]
                    batch_labels = data['train_labels'][offset:(offset + self.batch_size)]

                    feed_dict = {self.graph['x']: batch_data, self.graph['y_word']: batch_labels,
                                 self.graph['embeddings']: data['embedding'],
                                 self.graph['learning_rate']: learning_rate}

                    if dropout is not None:
                        feed_dict[self.graph['dropout']] = dropout

                    if self.init_trainable == False and zero_init_state == False:
                        if training_state_fw is not None:
                            feed_dict[self.graph['init_state_fw']] = training_state_fw
                        if training_state_bw is not None:
                            feed_dict[self.graph['init_state_bw']] = training_state_bw

                    training_loss_, training_state_fw, training_state_bw, _ = sess.run([self.graph['total_loss_word'],
                                                                                        self.graph['final_state_fw'],
                                                                                        self.graph['final_state_bw'],
                                                                                        self.graph['train_step_word']],
                                                                                       feed_dict)
                    training_loss += training_loss_

                training_losses.append(training_loss / steps_per_train_epoch)
                if verbose_graph:
                    ax_loss.plot(training_losses)
                    fig.canvas.draw()
                if verbose:
                    print("Average training loss for Epoch", epoch, ":", training_loss / steps_per_train_epoch)

                # ----------------- TESTING -------------------
                if 'test_data' in data.keys():
                    testing_loss = 0
                    testing_state_fw = None
                    testing_state_bw = None
                    for step in range(steps_per_test_epoch):
                        offset = (step * self.batch_size) % (data['test_data'].shape[0] - self.batch_size)
                        batch_data = data['test_data'][offset:(offset + self.batch_size)]
                        batch_labels = data['test_labels'][offset:(offset + self.batch_size)]
                        feed_dict = {self.graph['x']: batch_data, self.graph['y_word']: batch_labels,
                                     self.graph['embeddings']: data['embedding']}
                        if self.init_trainable == False and zero_init_state == False:
                            if testing_state_fw is not None:
                                feed_dict[self.graph['init_state_fw']] = testing_state_fw
                            if testing_state_bw is not None:
                                feed_dict[self.graph['init_state_bw']] = testing_state_bw
                        testing_loss_, testing_state_fw, testing_state_bw = sess.run([self.graph['total_loss_word'],
                                                                                      self.graph['final_state_fw'],
                                                                                      self.graph['final_state_bw']],
                                                                                     feed_dict)
                        testing_loss += testing_loss_

                    testing_losses.append(testing_loss / steps_per_test_epoch)
                    if verbose_graph:
                        ax_loss2.plot(testing_losses)
                        fig.canvas.draw()
                    if verbose:
                        print("Average  testing loss for Epoch", epoch, ":", testing_loss / steps_per_test_epoch)

            if isinstance(save, str):
                self.graph['saver'].save(sess, save)

            np.save("bidrect_base_training_losses.npy", training_losses)
            np.save("bidrect_base_testing_losses.npy", testing_losses)
        return dict(training_losses=training_losses, testing_losses=testing_losses)

    def train_finetune_model(self, data, num_epochs=0, learning_rate=1e-4, num_epochs2=None, learning_rate2=1e-4,
                             zero_init_state=True, dropout=1.0, dropout2=1.0, verbose=True, verbose_graph=False,
                             save=False, pretrain_model=None, training_resume=False, label_weight=None, debug=True):
        tf.set_random_seed(2345)
        num_classes = self.num_classes
        steps_per_train_epoch = int(data['train_data'].shape[0] / self.batch_size) + 1
        if 'test_data' in data.keys():
            steps_per_test_epoch = int(data['test_data'].shape[0] / self.batch_size) + 1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            testing_losses = []
            
            #For Method1
            training_acces = []
            testing_acces = []
            training_precisions = []
            testing_precisions = []
            training_recalls = []
            testing_recalls = []
            
            #For Method2
            training_acces2 = []
            testing_acces2 = []
            training_precisions2 = []
            testing_precisions2 = []
            training_recalls2 = []
            testing_recalls2 = []
            
            #For Method3
            training_acces3 = []
            testing_acces3 = []
            training_precisions3 = []
            testing_precisions3 = []
            training_recalls3 = []
            testing_recalls3 = []

            if pretrain_model is not None:
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_model))
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading Model ...")
                    self.graph["saver"].restore(sess, ckpt.model_checkpoint_path)
                    print("Pre-trained Model Loaded")
                    if training_resume:
                        if num_epochs2 is not None and num_epochs2 > 0:
                            if os.path.exists("bidrect_fine_training_losses.npy"):
                                training_losses = np.load("bidrect_fine_training_losses.npy").tolist()
                            if os.path.exists("bidrect_fine_testing_losses.npy"):
                                testing_losses = np.load("bidrect_fine_testing_losses.npy").tolist()
                            if os.path.exists("bidrect_fine_training_acces.npy"):
                                training_acces = np.load("bidrect_fine_training_acces.npy").tolist()
                            if os.path.exists("bidrect_fine_testing_acces.npy"):
                                testing_acces = np.load("bidrect_fine_testing_acces.npy").tolist()
                            if os.path.exists("bidrect_fine_training_precisions.npy"):
                                training_losses = np.load("bidrect_fine_training_precisions.npy").tolist()
                            if os.path.exists("bidrect_fine_testing_precisions.npy"):
                                testing_losses = np.load("bidrect_fine_testing_precisions.npy").tolist()
                            if os.path.exists("bidrect_fine_training_recalls.npy"):
                                training_acces = np.load("bidrect_fine_training_recalls.npy").tolist()
                            if os.path.exists("bidrect_fine_testing_recalls.npy"):
                                testing_acces = np.load("bidrect_fine_testing_recalls.npy").tolist()
                        else:
                            if os.path.exists("bidrect_retask_training_losses.npy"):
                                training_losses = np.load("bidrect_retask_training_losses.npy").tolist()
                            if os.path.exists("bidrect_retask_testing_losses.npy"):
                                testing_losses = np.load("bidrect_retask_testing_losses.npy").tolist()
                            if os.path.exists("bidrect_retask_training_acces.npy"):
                                training_acces = np.load("bidrect_retask_training_acces.npy").tolist()
                            if os.path.exists("bidrect_retask_testing_acces.npy"):
                                testing_acces = np.load("bidrect_retask_testing_acces.npy").tolist()
                            if os.path.exists("bidrect_retask_training_precisions.npy"):
                                training_losses = np.load("bidrect_retask_training_precisions.npy").tolist()
                            if os.path.exists("bidrect_retask_testing_precisions.npy"):
                                testing_losses = np.load("bidrect_retask_testing_precisions.npy").tolist()
                            if os.path.exists("bidrect_retask_training_recalls.npy"):
                                training_acces = np.load("bidrect_retask_training_recalls.npy").tolist()
                            if os.path.exists("bidrect_retask_testing_recalls.npy"):
                                testing_acces = np.load("bidrect_retask_testing_recalls.npy").tolist()

            print("training start ...--")

            if num_epochs2 is not None and num_epochs2 > 0:
                num_epochs_combine = num_epochs + num_epochs2
            else:
                num_epochs_combine = num_epochs

            if num_epochs_combine == 0:
                print("Num of epoch must be defined")
                return 0

            freeze_on = 100
            if verbose_graph:
                fig, (ax_loss, ax_acc, ax_precision, ax_recall) = plt.subplots(4, 1, figsize=(8, 8))
                ax_loss.set_ylabel("Cross Entropy")
                ax_loss.set_xlabel("Epoches")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_xlabel("Epoches")
                ax_precision.set_ylabel("Precision")
                ax_precision.set_xlabel("Epoches")
                ax_recall.set_ylabel("Recall")
                ax_recall.set_xlabel("Epoches")
                plt.ion()
                fig.show()
                fig.canvas.draw()

            total_loss_op = self.graph['total_loss_sentiment']
            pred_op = self.graph['preds_sentiment']
            learning_rate_decay = False
            learning_rate_changed = False
            
            for epoch in range(num_epochs_combine):
                training_loss = 0
                
                #For Method1
                training_acc_array = np.zeros(num_classes, dtype =np.float32)
                training_prec_array = np.zeros(num_classes, dtype =np.float32)
                training_recl_array = np.zeros(num_classes, dtype =np.float32)

                #For Method2
                training_match_array = np.zeros(num_classes, dtype =np.float32)
                training_tp_array = np.zeros(num_classes, dtype =np.float32)
                training_fp_array = np.zeros(num_classes, dtype =np.float32)
                training_fn_array = np.zeros(num_classes, dtype =np.float32)
                training_prec_batches_ignored = np.zeros(num_classes, dtype =np.float32)
                training_recl_batches_ignored = np.zeros(num_classes, dtype =np.float32)
                training_num_true_labels = np.zeros(num_classes, dtype =np.float32)
                training_num_samples = 0

                #For Method3
                training_acc3 = 0
                training_precision3 = 0
                training_recall3 = 0
                
                               
                training_state_fw = None
                training_state_bw = None

                if epoch < num_epochs:
                    train_op = self.graph['train_step_sentiment_freeze']
                    learning_rate_feed = learning_rate
                    dropout_feed = dropout
                else:
                    train_op = self.graph['train_step_sentiment_unfreeze']
                    learning_rate_feed = learning_rate2
                    if learning_rate_decay:
                        learning_rate_feed = learning_rate_feed / 10
                        learning_rate_decay = False
                        print("\33[102m\33[97m Learning Rate is decayed to: \33[0m\33[0m", learning_rate_feed)
                    dropout_feed = dropout2

                for step in range(steps_per_train_epoch):
                    offset = (step * self.batch_size) % (data['train_data'].shape[0] - self.batch_size)
                    batch_data = data['train_data'][offset:(offset + self.batch_size)]
                    batch_labels = data['train_labels'][offset:(offset + self.batch_size)]
                    if label_weight is None:
                        label_weight = np.diag(np.ones(self.num_classes))
                    
                    #print("label_weight: ",label_weight)
                     
                    feed_dict = {self.graph['x']: batch_data, self.graph['y_sentiment']: batch_labels,
                                 self.graph['embeddings']: data['embedding'],
                                 self.graph['label_weight']: label_weight,
                                 self.graph['learning_rate']: learning_rate_feed, self.graph['dropout']: dropout_feed}
                    if self.init_trainable == False and zero_init_state == False:
                        if training_state_fw is not None:
                            feed_dict[self.graph['init_state_fw']] = training_state_fw
                        if training_state_bw is not None:
                            feed_dict[self.graph['init_state_bw']] = training_state_bw
                    training_loss_, training_state_fw, training_state_bw, freeze_check_, pred_, _ = sess.run(
                        [total_loss_op,
                         self.graph['final_state_fw'],
                         self.graph['final_state_bw'],
                         self.graph['freeze_check'],
                         pred_op,
                         train_op],
                        feed_dict)

                    
                    pred_pd = np.array(pred_)
                    predictions = np.round(pred_pd)
                    labels = batch_labels  ## sw
                    
                    training_loss += training_loss_
                    
                    #Method 1
                    acc = np.zeros(num_classes, dtype =np.float32)
                    prec = np.zeros(num_classes, dtype =np.float32)
                    recl = np.zeros(num_classes, dtype =np.float32)

                    for clas in range(num_classes):
                        acc[clas] = metrics.accuracy_score(labels[:,clas], predictions[:,clas])
                        prec[clas] = metrics.precision_score(labels[:,clas], predictions[:,clas])
                        recl[clas] = metrics.recall_score(labels[:,clas], predictions[:,clas])

                    training_acc_array += acc
                    training_prec_array += prec
                    training_recl_array += recl
                    
                    #Method 2
                    match, true_positive, false_positive, false_negative, prec_batches_to_ignore, recl_batches_to_ignore = m_eval.true_false_positives_negatives(labels,predictions)
                                     
                    training_match_array += match
                    training_tp_array += true_positive
                    training_fp_array += false_positive
                    training_fn_array += false_negative
                    training_prec_batches_ignored += prec_batches_to_ignore
                    training_recl_batches_ignored += recl_batches_to_ignore
                    training_num_samples += predictions.shape[0]
                    training_num_true_labels += sum(labels)
                    
                    #Method 3
                    training_acc3 += m_eval.multilabel_accuracy(labels, predictions)
                    training_precision3 += m_eval.multilabel_precision(labels, predictions)
                    training_recall3 += m_eval.multilabel_recall(labels, predictions)
                    
                    if verbose:
                        if debug:
                            print(" ")
                            print("step:",step)
                            print("---------")

                            data_table = PrettyTable()
                            data_table.field_names = ["True_Labels", "Predictions"]
                            data_table.add_row([labels, predictions])
                            print(data_table)

                            step_table = PrettyTable()
                            step_table.field_names = ["Label", 
                                                       "Num_occ", 
                                                       "Matches", 
                                                       "Mthd1- Accuracy", 
                                                       "Mthd1- Precision", 
                                                       "Mthd1- Recall", 
                                                       "TP", 
                                                       "FP", 
                                                       "FN"]

                            for clas in range(num_classes):
                                step_table.add_row([clas, 
                                                     sum(labels)[clas], 
                                                     match[clas], 
                                                     round(acc[clas],3), 
                                                     round(prec[clas],3), 
                                                     round(recl[clas],3), 
                                                     true_positive[clas], 
                                                     false_positive[clas], 
                                                     false_negative[clas]])

                            print(step_table )
                        
                            print(" ")
                            print("Training accuracy3:", m_eval.multilabel_accuracy(labels, predictions))
                            print("Training precision3:", m_eval.multilabel_precision(labels, predictions))
                            print("Training recall3:", m_eval.multilabel_recall(labels, predictions))
                    
                    if step == 0 and epoch == 0:
                        base_freeze = freeze_check_
                    elif step == 1:
                        freeze_on = np.mean(np.square(np.array(freeze_check_) - np.array(base_freeze)))
                        
                steps_per_train_epoch_array = np.full(num_classes, steps_per_train_epoch)
               
                training_accuracy = training_acc_array/steps_per_train_epoch
                training_precision = training_prec_array/(steps_per_train_epoch - training_prec_batches_ignored)
                training_recall = training_recl_array/(steps_per_train_epoch - training_recl_batches_ignored)
                
                #Method 1
                training_losses.append(training_loss / steps_per_train_epoch)
                training_acces.append(training_accuracy)
                training_precisions.append(training_precision)
                training_recalls.append(training_recall)
                
                #Method 2
                training_accuracy2 = training_match_array/training_num_samples
                training_precision2 = training_tp_array/(training_tp_array + training_fp_array)
                training_recall2 = training_tp_array/(training_tp_array + training_fn_array)
                 
                training_acces2.append(training_accuracy2)
                training_precisions2.append(training_precision2)
                training_recalls2.append(training_recall2)
                
                #Method 3
                training_acces3.append(training_acc3/steps_per_train_epoch)
                training_precisions3.append(training_precision3 /steps_per_train_epoch)
                training_recalls3.append(training_recall3 / steps_per_train_epoch)
                
                                                   
                if verbose:
                    if epoch == 0 or epoch == num_epochs:
                        if freeze_on > 0 and epoch == num_epochs:
                            print("RNN Layer Freeze: \33[101m\33[97m DISENGAGE \33[0m\33[0m")
                        else:
                            print("RNN Layer Freeze: \33[102m\33[97m   ENGAGE  \33[0m\33[0m")
                            
                    
                    print(" ")
                    print("epoch: ",epoch, "Tranining")
                    print("=================")
                    
                    epoch_table = PrettyTable()
                    epoch_table.field_names = ["Label", 
                                               "Num_occur",
                                               "Acc", 
                                               "M1- Pre", 
                                               "M1- Rec",  
                                               "M2- Pre", 
                                               "M2- Rec", 
                                               "tp","fp","fn"]

                    for clas in range(num_classes):
                        epoch_table.add_row([clas, 
                                             round(training_num_true_labels[clas],0) ,
                                             round(training_accuracy[clas],3), 
                                             round(training_precision[clas],3), 
                                             round(training_recall[clas],3), 
                                             round(training_precision2[clas],3), 
                                             round(training_recall2[clas],3),
                                             training_tp_array[clas],
                                             training_fp_array[clas],
                                             training_fn_array[clas]])

                    print(epoch_table)
                    
                    print(" ")
                    print("Average training Accuracy :", training_acces3[-1])
                    print("Average training Precision :", training_precisions3[-1])
                    print("Average training Recall:", training_recalls3[-1])
                    
                if verbose_graph:
                    ax_acc.plot(training_acces, 'C1', label="Train Accuracy")
                    ax_precision.plot(training_precisions, 'C1', label="Train Precision")
                    ax_recall.plot(training_recalls, 'C1', label="Train Recall")
                    ax_loss.plot(training_losses, 'C1', label="Train Loss")
                    fig.canvas.draw()

                # ----------------- TESTING -------------------
                if 'test_data' in data.keys():
                    testing_loss = 0
                                        
                    #For Method1
                    testing_acc_array = np.zeros(num_classes, dtype =np.float32)
                    testing_prec_array = np.zeros(num_classes, dtype =np.float32)
                    testing_recl_array = np.zeros(num_classes, dtype =np.float32)

                    #For Method2
                    testing_match_array = np.zeros(num_classes, dtype =np.float32)
                    testing_tp_array = np.zeros(num_classes, dtype =np.float32)
                    testing_fp_array = np.zeros(num_classes, dtype =np.float32)
                    testing_fn_array = np.zeros(num_classes, dtype =np.float32)
                    testing_prec_batches_ignored = np.zeros(num_classes, dtype =np.float32)
                    testing_recl_batches_ignored = np.zeros(num_classes, dtype =np.float32)
                    testing_num_true_labels = np.zeros(num_classes, dtype =np.float32)
                    testing_num_samples = 0

                    #For Method3
                    testing_acc3 = 0
                    testing_precision3 = 0
                    testing_recall3 = 0                  
                    
                    testing_state_fw = None
                    testing_state_bw = None
                    
                    for step in range(steps_per_test_epoch):
                        offset = (step * self.batch_size) % (data['test_data'].shape[0] - self.batch_size)
                        batch_data = data['test_data'][offset:(offset + self.batch_size)]
                        batch_labels = data['test_labels'][offset:(offset + self.batch_size)]
                        if label_weight is None:
                            label_weight = np.diag(np.ones(self.num_classes))
                        feed_dict = {self.graph['x']: batch_data, 
                                     self.graph['y_sentiment']: batch_labels,
                                     self.graph['label_weight']: label_weight,
                                     self.graph['embeddings']: data['embedding']}
                        if self.init_trainable == False and zero_init_state == False:
                            if testing_state_fw is not None:
                                feed_dict[self.graph['init_state_fw']] = testing_state_fw
                            if testing_state_bw is not None:
                                feed_dict[self.graph['init_state_bw']] = testing_state_bw
                        testing_loss_, testing_state_fw, testing_state_bw, pred_ = sess.run([total_loss_op,
                                                                                             self.graph[
                                                                                                 'final_state_fw'],
                                                                                             self.graph[
                                                                                                 'final_state_bw'],
                                                                                             pred_op],
                                                                                            feed_dict)
                        pred_pd = np.array(pred_)
                        predictions = np.round(pred_pd)
                        labels = batch_labels  ## sw
                        
                        #Method 1
                        acc = np.zeros(num_classes, dtype =np.float32)
                        prec = np.zeros(num_classes, dtype =np.float32)
                        recl = np.zeros(num_classes, dtype =np.float32)

                        for clas in range(num_classes):
                            acc[clas] = metrics.accuracy_score(labels[:,clas], predictions[:,clas])
                            prec[clas] = metrics.precision_score(labels[:,clas], predictions[:,clas])
                            recl[clas] = metrics.recall_score(labels[:,clas], predictions[:,clas])

                        testing_acc_array += acc
                        testing_prec_array += prec
                        testing_recl_array += recl

                        #Method 2
                        match, true_positive, false_positive, false_negative, prec_batches_to_ignore, recl_batches_to_ignore = m_eval.true_false_positives_negatives(labels,predictions)

                        testing_match_array += match
                        testing_tp_array += true_positive
                        testing_fp_array += false_positive
                        testing_fn_array += false_negative
                        testing_prec_batches_ignored += prec_batches_to_ignore
                        testing_recl_batches_ignored += recl_batches_to_ignore
                        testing_num_samples += predictions.shape[0]
                        testing_num_true_labels += sum(labels)

                        #Method 3
                        testing_acc3 += m_eval.multilabel_accuracy(labels, predictions)
                        testing_precision3 += m_eval.multilabel_precision(labels, predictions)
                        testing_recall3 += m_eval.multilabel_recall(labels, predictions)
                        
                        if verbose:
                            if debug:
                                print(" ")
                                print("step:",step)
                                print("---------")

                                data_table = PrettyTable()
                                data_table.field_names = ["True_Labels", "Predictions"]
                                data_table.add_row([labels, predictions])
                                print(data_table)

                                step_table = PrettyTable()
                                step_table.field_names = ["Label", 
                                                           "Num_occ", 
                                                           "Matches", 
                                                           "Mthd1- Accuracy", 
                                                           "Mthd1- Precision", 
                                                           "Mthd1- Recall", 
                                                           "TP", 
                                                           "FP", 
                                                           "FN"]

                                for clas in range(num_classes):
                                    step_table.add_row([clas, 
                                                         sum(labels)[clas], 
                                                         match[clas], 
                                                         round(acc[clas],3), 
                                                         round(prec[clas],3), 
                                                         round(recl[clas],3), 
                                                         true_positive[clas], 
                                                         false_positive[clas], 
                                                         false_negative[clas]])

                                print(step_table )

                                print(" ")
                                print("Testing accuracy3:", m_eval.multilabel_accuracy(labels, predictions))
                                print("Testing precision3:", m_eval.multilabel_precision(labels, predictions))
                                print("Testing recall3:", m_eval.multilabel_recall(labels, predictions))

                    
                    
                    steps_per_test_epoch_array = np.full(num_classes, steps_per_test_epoch)
               
                    testing_accuracy = testing_acc_array/steps_per_test_epoch
                    testing_precision = testing_prec_array/(steps_per_test_epoch - testing_prec_batches_ignored)
                    testing_recall = testing_recl_array/(steps_per_test_epoch - testing_recl_batches_ignored)

                    #Method 1
                    testing_losses.append(testing_loss / steps_per_test_epoch)
                    testing_acces.append(testing_accuracy)
                    testing_precisions.append(testing_precision)
                    testing_recalls.append(testing_recall)

                    #Method 2
                    testing_accuracy2 = testing_match_array/testing_num_samples
                    testing_precision2 = testing_tp_array/(testing_tp_array + testing_fp_array)
                    testing_recall2 = testing_tp_array/(testing_tp_array + testing_fn_array)

                    testing_acces2.append(testing_accuracy2)
                    testing_precisions2.append(testing_precision2)
                    testing_recalls2.append(testing_recall2)

                    #Method 3
                    testing_acces3.append(testing_acc3/steps_per_test_epoch)
                    testing_precisions3.append(testing_precision3 /steps_per_test_epoch)
                    testing_recalls3.append(testing_recall3 / steps_per_test_epoch)
                    
                                        
                    if verbose:
                        print(" ")
                        print("epoch: ",epoch, " Testing")
                        print("*****************")

                        epoch_table = PrettyTable()
                        epoch_table.field_names = ["Label", 
                                                   "Num_occur",
                                                   "Acc", 
                                                   "M1 Pre", 
                                                   "M1- Rec", 
                                                   "M2- Pre", 
                                                   "M2- Rec",
                                                   "tp","fp","fn"]

                        for clas in range(num_classes):
                            epoch_table.add_row([clas, 
                                                 round(testing_num_true_labels[clas],0) ,
                                                 round(testing_accuracy[clas],3), 
                                                 round(testing_precision[clas],3), 
                                                 round(testing_recall[clas],3), 
                                                 round(testing_precision2[clas],3), 
                                                 round(testing_recall2[clas],3),
                                                 testing_tp_array[clas],
                                                 testing_fp_array[clas],
                                                 testing_fn_array[clas]])

                        print(epoch_table)

                        print(" ")
                        print("Average testing Accuracy :", testing_acces3[-1])
                        print("Average testing Precision :", testing_precisions3[-1])
                        print("Average testing Recall :", testing_recalls3[-1])
                        
                    if verbose_graph:
                        ax_acc.plot(testing_acces, 'C0', label="Validation Accuracy")
                        ax_precision.plot(testing_precisions, 'C0', label="Validation Precision")
                        ax_recall.plot(testing_recalls, 'C0', label="Validation Recall")
                        ax_loss.plot(testing_losses, 'C0', label="Validation Loss")
                        fig.canvas.draw()
                    #if num_epochs2 is None and testing_acces[-1] >= 0.78:  ## commented by SW
                    #    break  ## commented by SW
                    # if num_epochs2 is not None and testing_acces[-1] >= 0.79 and learning_rate_changed == False:
                    #    learning_rate_changed = True
                    #    learning_rate_decay = True
                    # if num_epochs2 is not None and testing_acces[-1] >= 0.795:
                    #    break

            if isinstance(save, str):
                self.graph['saver'].save(sess, save)
            if num_epochs2 is not None and num_epochs2 > 0:
                np.save("bidrect_fine_training_losses.npy", training_losses)
                np.save("bidrect_fine_testing_losses.npy", testing_losses)
                np.save("bidrect_fine_training_acces.npy", training_acces)
                np.save("bidrect_fine_testing_acces.npy", testing_acces)
                np.save("bidrect_fine_training_precisions.npy", training_precisions)
                np.save("bidrect_fine_testing_precisions.npy", testing_precisions)
                np.save("bidrect_fine_training_recalls.npy", training_recalls)
                np.save("bidrect_fine_testing_recalls.npy", testing_recalls)
            else:
                np.save("bidrect_retask_training_losses.npy", training_losses)
                np.save("bidrect_retask_testing_losses.npy", testing_losses)
                np.save("bidrect_retask_training_acces.npy", training_acces)
                np.save("bidrect_retask_testing_acces.npy", testing_acces)
                np.save("bidrect_retask_training_precisions.npy", training_precisions)
                np.save("bidrect_retask_testing_precisions.npy", testing_precisions)
                np.save("bidrect_retask_training_recalls.npy", training_recalls)
                np.save("bidrect_retask_testing_recalls.npy", testing_recalls)
            self.sess = sess
        return dict(training_losses=training_losses, testing_losses=testing_losses)

    def feedforward(self, data, model_path=None):
        if self.sess is None and model_path is None:
            print("Model has not been trained yet, No model path is provided either")
            return 0
        if model_path is not None:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_path))
            print("Loading Model ...")
            if ckpt and ckpt.model_checkpoint_path:
                self.sess = tf.Session()
                self.graph["saver"].restore(self.sess, ckpt.model_checkpoint_path)
                print("Model Loaded")
            else:
                print("Model Can Not Load")
                return 0

        preds = []
        batch_steps = int(data['data'].shape[0] / self.batch_size) + 1
        batch_remant = data['data'].shape[0] % self.batch_size
        for step in range(batch_steps):
            offset = (step * self.batch_size) % (data['data'].shape[0] - self.batch_size)
            batch_data = data['data'][offset:(offset + self.batch_size)]
            feed_dict = {self.graph['x']: batch_data, self.graph['embeddings']: data['embedding']}
            pred = self.sess.run(self.graph['preds_sentiment'], feed_dict)
            preds.append(pred)

        if batch_remant > 0:
            batch_data_tail = data['data'][-batch_remant:]
            batch_data_pad = np.zeros([self.batch_size - batch_remant, self.sequence_length], dtype='int32')
            batch_data = np.concatenate((batch_data_tail, batch_data_pad), axis=0)
            feed_dict = {self.graph['x']: batch_data, self.graph['embeddings']: data['embedding']}
            pred = self.sess.run(self.graph['preds_sentiment'], feed_dict)
            preds.append(pred[:batch_remant])

        output = np.concatenate(preds, 0)
        return output
