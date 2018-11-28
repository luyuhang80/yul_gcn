# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import graph
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import scipy.sparse
import numpy as np
import os, time, collections, shutil

class model():
    """
    Siamese Graph CNN which uses the Chebyshev approximation.
    Following the graph convolutional layers, the inner product of each node's features
    from the pair of graphs is used as input of the fully connected layer.
    Ktena et al., MICCAI 2017

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def  __init__(self, L0,L1, F0,F1, K, p, M, input_features0,input_features1, lamda, mu,
                num_epochs=20, learning_rate=0.01, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=50, eval_frequency=200,
                dir_name=''):
        
        self.regularizers = []
        self.input_features0 = input_features0
        self.input_features1 = input_features1
        self.lamda = lamda
        self.mu = mu
                # Verify the consistency w.r.t. the number of layers.
        #assert len(L) >= len(F) == len(K) == len(p)
        assert len(F0) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L0) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.

        # Keep the useful Laplacians only. May be zero.
        M_0 = L0[0].shape[0]
        j = 0
        self.L0 = []
        for pp in p:
            self.L0.append(L0[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L0 = self.L0
        L1 = L0
        M_1 = M_0

        # Store attributes and bind operations.
        self.L0,self.L1,self.F0, self.F1,self.K, self.p, self.M = L0,L1, F0,F1 ,K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name

        # Build the computational graph.
        self.build_graph(M_0,M_1)


    def loss(self, logits, labels,regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):

            # n_values = np.max(labels) + 1
            onehot_labels = tf.one_hot(labels,2)

            with tf.name_scope('cross_loss'):
                cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits))
            
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(onehot_labels,axis=1), tf.argmax(logits, axis=1)), tf.float32))

            loss = cross_loss + regularization

            return loss,accuracy

    def fit(self, train_data1,train_data2,train_labels, val_data1, val_data2,val_labels):
        # t_process, t_wall = time.process_time(), time.time()
        t_start = time.time()
        sess = tf.Session(graph=self.graph)
#        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
#        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        # ckpt=tf.train.get_checkpoint_state('./checkpoints/')
        # self.op_saver.restore(sess,ckpt.model_checkpoint_path)
        
        # Training.
        aucs = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data1.shape[0] / self.batch_size)

        for step in range(1, num_steps + 1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data1.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data1, batch_data2,batch_labels = train_data1[idx, :, :], train_data2[idx,:,:],train_labels[idx]
            if type(batch_data1) is not np.ndarray:
                batch_data1 = batch_data1.toarray()  # convert sparse matrices
            if type(batch_data2) is not np.ndarray:
                batch_data2 = batch_data2.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data1: batch_data1, self.ph_data2:batch_data2,self.ph_labels: batch_labels,
                         self.ph_dropout: self.dropout}
            #
            learning_rate = sess.run([self.op_train], feed_dict)
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data1.shape[0]
                # np.save('./Tsne/feat'+str(step)+'-'+str(step+self.batch_size)+'.npy')
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.4e}'.format(learning_rate[0]))
                string, auc, loss, scores_summary = self.evaluate(train_data1,train_data2,train_labels, sess)
                print('  training {}'.format(string))

                string, auc, loss, scores_summary = self.evaluate(val_data1,val_data2,val_labels, sess)
                print('  validation {}'.format(string))
                t_now = time.time()
                print('need time : {:.2f}h more , cost : {:.2f}h \n'.format(((t_now - t_start) * (num_steps + 1 - step)/step)/3600, (t_now - t_start)/3600))
                # print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

                aucs.append(auc)
                losses.append(loss)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(aucs), np.mean(aucs[-10:])))
        writer.close()
        sess.close()
        
        return aucs, losses

    def build_graph(self, M_0,M_1):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data1 = tf.placeholder(tf.float32, (self.batch_size, M_0,self.input_features0), 'data1')
                self.ph_data2 = tf.placeholder(tf.float32, (self.batch_size, M_1, self.input_features1), 'data2')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits  = self.inference(self.ph_data1,self.ph_data2,self.ph_dropout)

            self.op_loss,self.accuracy = self.loss(op_logits,self.ph_labels,self.regularization)

            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)
            
            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Save for model parameters.
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def gcn(self, x, L, Fout, K, regularization=False):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_],0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=regularization)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def corr_layer(self, x1, x2):
        N1, M1, F1 = x1.get_shape()
        N2, M2, F2 = x1.get_shape()
        x1 = tf.reshape(x1, [int(N1 * M1), int(F1)])
        x2 = tf.reshape(x2, [int(N2 * M2), int(F2)])
        # multiply ->  yuan su xiang cheng
        # reduce_sum -> an weidu ya bian
        corr = tf.reduce_sum(tf.multiply(x1, x2), 1, keep_dims=True)
        # print(type(corr))
        res = tf.reshape(corr, [int(N1), int(M1), 1])
        return res

    def build_model(self, g):
        # Graph convolutional layers.
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i + 1)):
                with tf.name_scope('gcn'):
                    g = self.gcn(g, self.L0[i] ,self.F0[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    g = self.brelu(g)
                with tf.name_scope('pooling'):
                    g = self.pool(g, self.p[i])
        return g
        
    def _inference(self, x_0, x_1,dropout):

        # Share weights between the two models of the pair
        with tf.variable_scope("siamese") as scope:
            # m_0 = self.build_model(x_0)
            m_0 = x_0
            m_1 = x_1
            # scope.reuse_variables()
            # m_1 = self.build_model(x_1)

        N1,M1,F1 = m_0.get_shape()
        N2,M2,F2 = m_1.get_shape()
        model_0 = tf.reshape(m_0,[int(N1),int(M1*F1)])
        model_1 = tf.reshape(m_1,[int(N2),int(M2*F2)])

        fc_layers = 1024
        with tf.variable_scope('txt0_fc'):
            model_0 = self.fc(model_0,fc_layers,relu=False)
        with tf.variable_scope('txt1_fc'):
            model_1 = self.fc(model_1,fc_layers,relu=False)

        # dot
        x = tf.multiply(model_0,model_1)
        
        # Logits linear layer
        with tf.variable_scope('logits'):
            x = tf.nn.dropout(x, dropout)
            x = tf.nn.softmax(self.fc(x, 2 ,relu=False))

        return x # tf.sigmoid(x)

    def inference(self, data1,data2,dropout):
        logits = self._inference(data1, data2,dropout)
        return logits 

    def predict(self, data1,data2,labels=None, sess=None):
        loss,acc = 0,0
        size = data1.shape[0]
        predictions = np.empty([size,2])
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data1 = np.zeros((self.batch_size, data1.shape[1], data1.shape[2]))
            batch_data2 = np.zeros((self.batch_size, data2.shape[1], data2.shape[2]))

            tmp_data1 = data1[begin:end, :, :]
            tmp_data2 = data2[begin:end, :, :]
            if type(tmp_data1) is not np.ndarray:
                tmp_data1 = tmp_data1.toarray()  # convert sparse matrices
            if type(tmp_data2) is not np.ndarray:
                tmp_data2 = tmp_data2.toarray()  # convert sparse matrices
            batch_data1[:end - begin] = tmp_data1
            batch_data2[:end - begin] = tmp_data2

            feed_dict = {self.ph_data1: batch_data1, self.ph_data2:batch_data2,self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros([self.batch_size])
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss, accuracy = sess.run([self.op_prediction,self.op_loss,self.accuracy], feed_dict)
                loss += batch_loss
                acc += accuracy

            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end,:] = batch_pred[:end - begin,:]

        if labels is not None:
            return predictions, loss * self.batch_size / size, acc * self.batch_size / size
        else:
            return predictions


    def evaluate(self, data1,data2 ,labels, sess=None):
        scores, loss , acc = self.predict(data1,data2 ,labels, sess)
        string = 'samples: {}, ACC : {:.2f}, loss: {:.4e}'.format(len(labels), acc, loss)
        return string, acc, loss, scores

    def prediction(self, logits):
        with tf.name_scope('prediction'):
            prediction = logits
            return prediction

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
                tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape,regularization=True):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)

        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        #initial = tf.constant_initializer(0.1)
        initial = tf.constant_initializer(0.0)

        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            tf.summary.histogram(var.op.name, var)
        return var
    
    def brelu(self, x,relu=True):
        """Bias and ReLU (if relu=True). One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        x = x + b
        return tf.nn.relu(x) if relu else x

    def pool(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x
    
    def fc(self, x, Mout,relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout],regularization=True)
        b = self._bias_variable([Mout],regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x
