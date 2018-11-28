# Copyright (c) 2018 Yuhang Lu <luyuhang@iie.ac.cn>

from lib import model, utils,graph
import numpy as np
import os
import time
from scipy import sparse
import random
#GPU 控制
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

print('start prepairing data')
x0_train,x1_train,y_train,x0_test,x1_test,y_test = utils.prepair_data()

# save and load data
#utils.save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test)
x0_train,x1_train,y_train,x0_test,x1_test,y_test = utils.load_data()

print('start build graph')
# Calculate Laplacians
g0=sparse.csr_matrix(utils.build_graph('./data/content_10_knn_graph.txt')).astype(np.float32)
print('graph_size:',g0.shape)
graphs0 = []
for i in range(3):
    graphs0.append(g0)
L0 = [graph.laplacian(A, normalized=True) for A in graphs0]
L1 = 1

# Graph Conv-net
f0,f1,features,K=1,1,1,3
params = dict()
params['num_epochs']     = 50
params['batch_size']     = 256
params['eval_frequency'] = int(50)
# params['eval_frequency'] = int(x1_train.shape[0] / (params['batch_size'] * 4))
# Architecture.
params['F0']              = [1,1]   # Number of graph convolutional filters.
params['F1']              = [1,1]   # Number of graph convolutional filters.
params['K']              = [K,K]   # Polynomial orders.
params['p']              = [1,1]    # Pooling sizes.
params['M']              = [1]    # Output dimensionality of fully connected layers.
params['input_features0'] = f0
params['input_features1'] = f1
params['lamda']          = 0.35
params['mu']             = 0.8
# Optimization.
params['regularization'] = 5e-3
params['dropout']        = 0.8
params['learning_rate']  = 1e-3
params['decay_rate']     = 0.95
params['momentum']       = 0
params['decay_steps']    = int(x1_train.shape[0] / params['batch_size'])

params['dir_name']       = 'siamese_' + time.strftime("%Y_%m_%d_%H_%M") + '_state'

# print(params)
print('start run model')
# Run model
sm_gcn = model.model(L0,L1, **params)

# train code
accuracy, loss = sm_gcn.fit(x0_train, x1_train,y_train, x0_test, x1_test,y_test)

