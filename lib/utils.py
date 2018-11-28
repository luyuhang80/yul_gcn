import os
import numpy as np
import time
import random
from random import choice
from lib import models_siamese,graph
from scipy.sparse import csr_matrix

# from sklearn.cross_validation import train_test_split

def build_graph(filename):
    start = time.clock()
    file = open(filename).read().strip().split('\n')
    graph=np.zeros((len(file),len(file)),dtype=np.int)
    y = 0
    for line in file:
        new_col=np.array(line.split())
        for i in new_col:
            graph[y][int(i)]=1
        y += 1
    end=time.clock()
    print('build txt graph cost time:{:.2f}s'.format(end-start))
    return graph

def load_wl(filename):
    dic,now = {},0
    for line in open('./data/'+filename):
        dic[line.strip().split()[0]] = now
        now += 1
    return dic

def get_list(filename):
    res = []
    for line in open(filename):
        tmp = line.strip()
        if tmp:
            res.append(tmp)
    return res

def build_bow(data,g):
    res = []
    for i,line in enumerate(data):
        # print('build bow ',i,'/',len(data))
        tmp = np.zeros([len(g)])
        count = 0
        for w in line.split():
            if w in g:
                tmp[g[w]] += 1
                count +=1
        # print(count)
        res.append(tmp)
    return np.expand_dims(np.array(res),2)

def prepair_data():

    g_content = load_wl('content_10_word_list.txt')

    x0_train_list = get_list('./data/train_cut_content_0.txt')
    x1_train_list = get_list('./data/train_cut_content_1.txt')
    x0_train = build_bow(x0_train_list,g_content)
    x1_train = build_bow(x1_train_list,g_content)

    y_train = np.array(get_list('./data/train_label.txt')).astype(int)

    x0_test_list = get_list('./data/test_cut_content_0.txt')
    x1_test_list = get_list('./data/test_cut_content_1.txt')
    x0_test = build_bow(x0_test_list,g_content)
    x1_test = build_bow(x1_test_list,g_content)

    y_test = np.array(get_list('./data/test_label.txt')).astype(int)

    return x0_train,x1_train,y_train,x0_test,x1_test,y_test

def save_data(x0_train,x1_train,y_train,x0_test,x1_test,y_test):
    np.save('./data/x0_train.npy',x0_train)
    np.save('./data/x1_train.npy',x1_train)
    np.save('./data/y_train.npy',y_train)
    np.save('./data/x0_test.npy',x0_test)
    np.save('./data/x1_test.npy',x1_test)
    np.save('./data/y_test.npy',y_test)
    
def load_data():
    x0_train = np.load('./data/x0_train.npy')
    x1_train = np.load('./data/x1_train.npy')
    y_train = np.load('./data/y_train.npy')
    x0_test = np.load('./data/x0_test.npy')
    x1_test = np.load('./data/x1_test.npy')
    y_test = np.load('./data/y_test.npy')
    return x0_train,x1_train,y_train,x0_test,x1_test,y_test
