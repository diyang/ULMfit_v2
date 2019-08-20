import os
import numpy as np
import pandas as pd
import tensorflow as tf
from customLSTMcell import LayerNormalizedLSTMCell
import matplotlib.pyplot as plt
import math
from sklearn import metrics

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

class RNN_build_graph:
    def __init__(self, state_size,
        num_words,
        num_classes,
        batch_size,
        sequence_length,
        cell_type = None,
        num_layers = 3,
        init_trainable = False):

        self.state_size = state_size
        self.num_words = num_words
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.init_trainable = init_trainable
        self.graph = self.build_graph()

    def get_graph(self):
        return self.graph

    def build_graph(self):
        reset_graph()
        with tf.variable_scope("rnn_layer"):
            x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name='input_placeholder')
            y_word = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name='labels_placeholder')
            y_sentiment = tf.placeholder(tf.int32, [self.batch_size], name='sentiments_placeholder')
            embeddings = tf.placeholder(tf.float32, [self.num_words, self.state_size], name='embeddings_placeholder')

            tf_dropout = tf.constant(1.0)
            tf_learning_rate = tf.constant(1e-4)

            rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

            if self.cell_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(self.state_size)
            elif self.cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
            elif self.cell_type == 'LN_LSTM':
                cell = LayerNormalizedLSTMCell(self.state_size)
            else:
                cell = tf.nn.rnn_cell.BasicRNNCell(self.state_size)

            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=tf_dropout)

            if self.cell_type == 'LSTM' or self.cell_type == 'LN_LSTM':
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
            else:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=tf_dropout)

            if self.init_trainable:
                init_state = []
                if self.cell_type == 'LSTM' or self.cell_type == 'LN_LSTM':
                    for rnn_layer in range(self.num_layers):
                        init_c = tf.dtypes.cast(tf.get_variable('init_'+str(rnn_layer)+'_c', [self.batch_size, self.state_size], trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_h = tf.dtypes.cast(tf.get_variable('init_'+str(rnn_layer)+'_h', [self.batch_size, self.state_size], trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_state.append(tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h))
                else:
                    for rnn_layer in range(self.num_layers):
                        init_s = tf.dtypes.cast(tf.get_variable('init_'+str(rnn_layer)+'_s', [self.batch_size, self.state_size], trainable=True, initializer=tf.constant_initializer(0.0)), dtype=tf.float32)
                        init_state.append(init_s)
                init_state = tuple(init_state)
            else:
                init_state = cell.zero_state(self.batch_size, tf.float32)

            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            rnn_outputs = tf.identity(rnn_outputs, 'rnn_outputs_tensor')

        #-------------------------------- NEXT WORDING PREDICTION --------------------------------------------
        #global_step = tf.Variable(0, trainable=False)
        #learning_rate_schedule = tf.train.cosine_decay(self.learning_rate, global_step, 5)
        with tf.variable_scope('word_prediction'):
            W = tf.get_variable('W', [self.state_size, self.num_words])
            b = tf.get_variable('b', [self.num_words], initializer=tf.constant_initializer(0.0))
            rnn_outputs_reshape = tf.reshape(rnn_outputs, [-1, self.state_size])
            word_logits = tf.matmul(rnn_outputs_reshape, W) + b
            word_predictions = tf.nn.softmax(word_logits, name='predictions')
            y_reshaped = tf.reshape(y_word, [-1])
            word_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_reshaped, logits=word_logits, name="word_cross_entropy")
            word_cross_entropy_mean = tf.reduce_mean(word_cross_entropy, name="word_cross_entropy_mean")
            word_train_step = tf.train.AdamOptimizer(tf_learning_rate).minimize(word_cross_entropy_mean)

        #------------------------------- SENTIMENT CLASSIFICATION --------------------------------------------
        with tf.variable_scope('sentiment_softmax'):
            # Hierarchical Attention Layer
            W_omega = tf.get_variable('W_omega', [self.state_size, self.state_size])
            b_omega = tf.get_variable('b_omega', [self.state_size])
            u_omega = tf.get_variable('u_omega', [self.state_size])
            v = tf.tanh(tf.tensordot(rnn_outputs, W_omega, axes=1)+b_omega)
            vu = tf.tensordot(v, u_omega, axes=1)
            alphas = tf.nn.softmax(vu)
            rnn_outputs_attn = tf.reduce_sum(rnn_outputs * tf.expand_dims(alphas, -1), 1)
            # Dense Layers
            rnn_outputs_lst = tf.squeeze(tf.slice(rnn_outputs, [0, (self.sequence_length-1), 0], [self.batch_size, 1, self.state_size]))
            rnn_outputs_concat = tf.concat([rnn_outputs_lst, rnn_outputs_attn], 1)
            W_s1 = tf.get_variable('W_s1', [self.state_size*2, self.state_size])
            b_s1 = tf.get_variable('b_s1', [self.state_size], initializer=tf.constant_initializer(0.0))
            W_s2 = tf.get_variable('W_s2', [self.state_size, self.num_classes])
            b_s2 = tf.get_variable('b_s2', [self.num_classes], initializer=tf.constant_initializer(0.0))
            fc1 = tf.matmul(rnn_outputs_concat, W_s1) + b_s1
            sentiment_logits = tf.matmul(fc1, W_s2) + b_s2
            sentiment_predictions = tf.nn.softmax(sentiment_logits, name='predictions_sentiment')
            sentiment_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_sentiment, logits=sentiment_predictions, name="cross_entropy_sentiment")
            sentiment_cross_entropy_mean = tf.reduce_mean(sentiment_cross_entropy, name="cross_entropy_sentiment_mean")

        freeze_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sentiment_softmax")
        sentiment_train_step_freeze = tf.train.AdamOptimizer(tf_learning_rate).minimize(sentiment_cross_entropy_mean, var_list=freeze_train_vars)
        sentiment_train_step_unfreeze = tf.train.AdamOptimizer(tf_learning_rate).minimize(sentiment_cross_entropy_mean)

        return dict(
            x = x,
            y_word = y_word,
            y_sentiment = y_sentiment,
            embeddings = embeddings,
            dropout = tf_dropout,
            learning_rate = tf_learning_rate,

            init_state = init_state,
            final_state = final_state,

            total_loss_word = word_cross_entropy_mean,
            train_step_word = word_train_step,
            preds_word = word_predictions,

            total_loss_sentiment = sentiment_cross_entropy_mean,
            train_step_sentiment_freeze = sentiment_train_step_freeze,
            train_step_sentiment_unfreeze = sentiment_train_step_unfreeze,
            preds_sentiment = sentiment_predictions,

            freeze_check = rnn_outputs_reshape,
            saver = tf.train.Saver()
        )

    def train_base_model(self, data, num_epochs, learning_rate, zero_init_state=True, verbose=True, verbose_graph=False, save=False, pretrain_model=None, dropout=None):
        tf.set_random_seed(2345)
        steps_per_train_epoch = int(data['train_data'].shape[0]/self.batch_size)+1
        if 'test_data' in data.keys():
            steps_per_test_epoch = int(data['test_data'].shape[0]/self.batch_size)+1

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
                    if os.path.exists("base_training_losses.npy"):
                        training_losses = np.load("base_training_losses.npy").tolist()
                    if os.path.exists("base_testing_losses.npy"):
                        testing_losses = np.load("base_testing_losses.npy").tolist()

            print("training start ...")

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
                training_loss = 0
                training_state = None
                for step in range(steps_per_train_epoch):
                    offset = (step*self.batch_size) % (data['train_data'].shape[0]-self.batch_size)
                    batch_data = data['train_data'][offset:(offset+self.batch_size)]
                    batch_labels = data['train_labels'][offset:(offset+self.batch_size)]

                    feed_dict={self.graph['x']: batch_data, self.graph['y_word']: batch_labels, self.graph['embeddings']: data['embedding'], self.graph['learning_rate']: learning_rate}
                    if dropout is not None:
                        feed_dict[self.graph['dropout']] = dropout

                    if self.init_trainable == False and zero_init_state == False:
                        if training_state is not None:
                            feed_dict[self.graph['init_state']] = training_state

                    training_loss_, training_state, _ = sess.run([self.graph['total_loss_word'],
                              self.graph['final_state'],
                              self.graph['train_step_word']],
                             feed_dict)
                    training_loss += training_loss_

                training_losses.append(training_loss/steps_per_train_epoch)
                if verbose_graph:
                    ax_loss.plot(training_losses)
                    fig.canvas.draw()
                if verbose:
                    print("Average training loss for Epoch", epoch, ":", training_loss/steps_per_train_epoch)

                #----------------- TESTING -------------------
                if 'test_data' in data.keys():
                    testing_loss = 0
                    testing_state = None
                    for step in range(steps_per_test_epoch):
                        offset = (step*self.batch_size) % (data['test_data'].shape[0]-self.batch_size)
                        batch_data = data['test_data'][offset:(offset+self.batch_size)]
                        batch_labels = data['test_labels'][offset:(offset+self.batch_size)]
                        feed_dict={self.graph['x']: batch_data, self.graph['y_word']: batch_labels, self.graph['embeddings']: data['embedding']}
                        if self.init_trainable == False and zero_init_state == False:
                            if testing_state is not None:
                                feed_dict[self.graph['init_state']] = testing_state
                        testing_loss_, testing_state = sess.run([self.graph['total_loss_word'],
                                                                 self.graph['final_state']],
                                                                feed_dict)
                        testing_loss += testing_loss_

                    testing_losses.append(testing_loss/steps_per_test_epoch)
                    if verbose_graph:
                        ax_loss2.plot(testing_losses)
                        fig.canvas.draw()
                    if verbose:
                        print("Average  testing loss for Epoch", epoch, ":", testing_loss/steps_per_test_epoch)

            if isinstance(save, str):
                self.graph['saver'].save(sess, save, global_step=(epoch+1))

            np.save("base_training_losses.npy", training_losses)
            np.save("base_testing_losses.npy", testing_losses)
        return dict(training_losses = training_losses, testing_losses=testing_losses)

    def train_finetune_model(self, data, num_epochs, learning_rate, num_epochs2=None, learning_rate2=1e-4, zero_init_state=True, dropout=None, dropout2=None, verbose=True, verbose_graph=False, save=False, pretrain_model=None):
        tf.set_random_seed(2345)
        steps_per_train_epoch = int(data['train_data'].shape[0]/self.batch_size)+1
        if 'test_data' in data.keys():
            steps_per_test_epoch = int(data['test_data'].shape[0]/self.batch_size)+1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            testing_losses = []
            training_acces = []
            testing_acces = []
            if pretrain_model is not None:
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_model))
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading Model ...")
                    self.graph["saver"].restore(sess, ckpt.model_checkpoint_path)
                    print("Pre-trained Model Loaded")

            print("training start ...")

            if num_epochs2 is not None and num_epochs2 > 0:
                num_epochs_combine = num_epochs+num_epochs2
            else:
                num_epochs_combine = num_epochs
            freeze_on = 100

            if verbose_graph:
                fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 8))
                ax_loss.set_ylabel("Cross Entropy")
                ax_loss.set_xlabel("Epoches")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_xlabel("Epoches")
                plt.ion()
                fig.show()
                fig.canvas.draw()

            total_loss_op = self.graph['total_loss_sentiment']
            pred_op = self.graph['preds_sentiment']
            for epoch in range(num_epochs_combine):
                training_loss = 0
                training_acc = 0
                training_state = None
                for step in range(steps_per_train_epoch):
                    offset = (step*self.batch_size) % (data['train_data'].shape[0]-self.batch_size)
                    batch_data = data['train_data'][offset:(offset+self.batch_size)]
                    batch_labels = data['train_labels'][offset:(offset+self.batch_size)]

                    if epoch < num_epochs:
                        train_op = self.graph['train_step_sentiment_freeze']
                        learning_rate_feed = learning_rate
                        dropout_feed = dropout
                    else:
                        train_op = self.graph['train_step_sentiment_unfreeze']
                        learning_rate_feed = learning_rate2
                        dropout_feed = dropout2

                    feed_dict={self.graph['x']: batch_data, self.graph['y_sentiment']: batch_labels, self.graph['embeddings']: data['embedding'], self.graph['learning_rate']: learning_rate_feed}
                    if dropout_feed is not None:
                        feed_dict[self.graph['dropout']] = dropout_feed
                    if self.init_trainable == False and zero_init_state == False:
                        if training_state is not None:
                            feed_dict[self.graph['init_state']] = training_state

                    training_loss_, training_state, freeze_check_, pred_, _ = sess.run([total_loss_op,
                                                                                        self.graph['final_state'],
                                                                                        self.graph['freeze_check'],
                                                                                        pred_op,
                                                                                        train_op],
                                                                                       feed_dict)

                    pred_pd = pd.DataFrame(pred_)
                    predictions = list(pred_pd.idxmax(axis=1))
                    labels = list(batch_labels)
                    training_acc += metrics.accuracy_score(labels, predictions)
                    training_loss += training_loss_

                    if step == 0 and epoch == 0:
                        base_freeze = freeze_check_
                    elif step == 1:
                        freeze_on = np.mean(np.square(np.array(freeze_check_) - np.array(base_freeze)))

                training_losses.append(training_loss/steps_per_train_epoch)
                training_acces.append(training_acc/steps_per_train_epoch)
                if verbose:
                    if epoch == 0 or epoch == num_epochs:
                        if freeze_on > 0 and epoch == num_epochs:
                            print("RNN Layer Freeze: \33[101m\33[97m DISENGAGE \33[0m\33[0m")
                        else:
                            print("RNN Layer Freeze: \33[102m\33[97m   ENGAGE  \33[0m\33[0m")
                    print("Average training loss for Epoch", epoch, ":", training_losses[-1])

                if verbose_graph:
                    ax_acc.plot(training_acces, 'C1', label="Train Accuracy")
                    ax_loss.plot(training_losses, 'C1', label="Train Loss")
                    fig.canvas.draw()

                #----------------- TESTING -------------------
                if 'test_data' in data.keys():
                    testing_loss = 0
                    testing_acc = 0
                    testing_state = None
                    for step in range(steps_per_test_epoch):
                        offset = (step*self.batch_size) % (data['test_data'].shape[0]-self.batch_size)
                        batch_data = data['test_data'][offset:(offset+self.batch_size)]
                        batch_labels = data['test_labels'][offset:(offset+self.batch_size)]

                        feed_dict={self.graph['x']: batch_data, self.graph['y_sentiment']: batch_labels, self.graph['embeddings']: data['embedding']}
                        if self.init_trainable == False and zero_init_state == False:
                            if testing_state is not None:
                                feed_dict[self.graph['init_state']] = testing_state
                        testing_loss_, testing_state, pred_ = sess.run([total_loss_op,
                                                                        self.graph['final_state'],
                                                                        pred_op],
                                                                       feed_dict)
                        pred_pd = pd.DataFrame(pred_)
                        predictions = list(pred_pd.idxmax(axis=1))
                        labels = list(batch_labels)
                        testing_acc += metrics.accuracy_score(labels, predictions)
                        testing_loss += testing_loss_

                    testing_acces.append(testing_acc/steps_per_test_epoch)
                    testing_losses.append(testing_loss/steps_per_test_epoch)
                    if verbose:
                        print("Average  testing loss for Epoch", epoch, ":", testing_losses[-1])
                    if verbose_graph:
                        ax_acc.plot(testing_acces, 'C0', label="Validation Accuracy")
                        ax_loss.plot(testing_losses, 'C0', label="Validation Loss")
                        fig.canvas.draw()

            if isinstance(save, str):
                self.graph['saver'].save(sess, save, global_step=(epoch+1))
            if num_epochs2 is not None and num_epochs2 > 0:
                np.save("fine_training_losses.npy", training_losses)
                np.save("fine_testing_losses.npy", testing_losses)
                np.save("fine_training_acces.npy", training_acces)
                np.save("fine_testing_acces.npy", testing_acces)
            else:
                np.save("retask_training_losses.npy", training_losses)
                np.save("retask_testing_losses.npy", testing_losses)
                np.save("retask_training_acces.npy", training_acces)
                np.save("retask_testing_acces.npy", testing_acces)
            self.sess = sess
        return dict(training_losses = training_losses, testing_losses=testing_losses)

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
        batch_steps = int(data['data'].shape[0]/self.batch_size)+1
        batch_remant = data['data'].shape[0]%self.batch_size
        for step in range(batch_steps):
            offset = (step*self.batch_size) % (data['data'].shape[0]-self.batch_size)
            batch_data = data['data'][offset:(offset+self.batch_size)]
            feed_dict={self.graph['x']: batch_data, self.graph['embeddings']: data['embedding']}
            pred = self.sess.run(self.graph['preds_sentiment'],feed_dict)
            preds.append(pred)

        if batch_remant > 0:
            batch_data_tail = data['data'][-batch_remant:]
            batch_data_pad = np.zeros([self.batch_size-batch_remant, self.sequence_length], dtype='int32')
            batch_data = np.concatenate((batch_data_tail, batch_data_pad), axis=0)
            feed_dict={self.graph['x']: batch_data, self.graph['embeddings']: data['embedding']}
            pred = self.sess.run(self.graph['preds_sentiment'],feed_dict)
            preds.append(pred[:batch_remant])

        output = np.concatenate(preds,0)
        return output
