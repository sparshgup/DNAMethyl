#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 26 2021

@author: sparshgupta
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FixedLocator
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
import tensorflow.compat.v1 as tf
import tensorflow.keras as tfnew
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.regularizers import l1, l2
#from sklearn.metrics import mean_squared_error as mse
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_absolute_error as mae
import keras


#Run Settings
dataset_name = 'dnamethyl'

#Model Settings
input_size = 100
output_size = 1
E_node = 64
A_node = 2
set_seed = 1
L_node = 64
moving_average_decay = 0.001

# Training & Optimizer
regularization_rate = 0.9
learning_rate_base = 0.8
learning_rate_decay = 0.99
batch_size = 128
train_step = 10

#Utils

class BatchCreate(object):
    
    def __init__(self,images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
        
    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        '''
        Disruption in the first epoch
        '''
        if self._epochs_completed ==0 and start ==0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
            
        if start+batch_size>self._num_examples:
            #finished epoch
            self._epochs_completed += 1
            '''
            When the remaining sample number of an epoch is less than batch size,
            the difference between them is calculated.
            '''
            rest_num_examples = self._num_examples-start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            
            '''Disrupt the data'''
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            '''next epoch'''
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part),axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
        
def show_result(ac_score_list,dataset_name):
    
    plot_spc = [5, 5]
    left_spc = 1
    len_params = 100

    use_arr = ac_score_list[left_spc:int((len_params - plot_spc[0]) / plot_spc[1])]
    ax = plt.subplot()
    ax.plot(np.arange(plot_spc[0] + plot_spc[1] * left_spc, len_params, plot_spc[1]), use_arr, '-o',
            label='AFS')
    plt.title(dataset_name)
    plt.ylabel('MAE')
    plt.xlabel('N')
    plt.ylim(0, 50)
    xmajorLocator = FixedLocator([5, 25, 50, 75, 100])
    xminorLocator = MultipleLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.grid(True, which='minor')

    ymajorLocator = MultipleLocator(5)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.grid(True, which='major')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])

    plt.show()


#Model 
    

def build(total_batch):
    X = tf.placeholder(tf.float32, [None, input_size])
    Y = tf.placeholder(tf.float32, [None, output_size])
    global_step = tf.Variable(0, trainable=False)

    tf.add_to_collection('input', X)
    tf.add_to_collection('output',Y)
    
    with tf.variable_scope('attention_module') as scope:
        
        E_W = tf.Variable(
            tf.truncated_normal([input_size, E_node], stddev = 0.1, seed = set_seed))
        
        E_b = tf.Variable(
            tf.constant(0.1, shape = [E_node]))

        E = tf.nn.tanh(tf.matmul(X,E_W)+ E_b)

        A_W = tf.Variable(
            tf.truncated_normal([input_size, E_node, A_node], stddev = 0.1, seed = set_seed))
        
        A_b = tf.Variable(
            tf.constant(0.1, shape=[input_size, A_node]))

        A_W_unstack = tf.unstack(A_W, axis=0)
        A_b_unstack = tf.unstack(A_b, axis=0)

        attention_out_list = []
        
        for i in range(input_size):
            
            attention_FC = tf.matmul(E, A_W_unstack[i]) + A_b_unstack[i]
            
            attention_out = tf.nn.softmax(attention_FC)

            attention_out = tf.expand_dims(attention_out[:,1], axis=1)

            attention_out_list.append(attention_out)
            
        A = tf.squeeze(tf.stack(attention_out_list, axis=1),axis=2)


    with tf.variable_scope("learning_module") as scope:
        
        
        G = tf.multiply(X, A)

        L_W1 = tf.Variable(
            tf.truncated_normal([input_size, L_node], stddev=0.1, seed = set_seed))
        
        L_b1 = tf.Variable(tf.constant(0.1, shape=[L_node]))
        
        L_W2 = tf.Variable(
            tf.truncated_normal([L_node, output_size], stddev=0.1, seed = set_seed))
        
        L_b2 = tf.Variable(tf.constant(0.1, shape=[output_size]))

     
        variable_averages = tf.train.ExponentialMovingAverage(
                moving_average_decay, global_step)
        
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())
        
        L_FC = tf.nn.relu(tf.matmul(G, L_W1) + L_b1)

        O = tf.matmul(L_FC, L_W2) + L_b2

        average_L_FC = tf.nn.relu(tf.matmul(G, variable_averages.average(L_W1)) + variable_averages.average(L_b1))
        average_O = tf.matmul(average_L_FC, variable_averages.average(L_W2)) + variable_averages.average(L_b2)
        
        
    with tf.name_scope("Loss") as scope:
        
        regularizer = tfnew.regularizers.l2(
                regularization_rate)
        
        regularization = regularizer(L_W1) + regularizer(L_W2)
        
        learning_rate = tf.train.exponential_decay(
                learning_rate_base, global_step, total_batch,
                learning_rate_decay)
        
        mse = tf.square(average_O - Y)
        
        mse_mean = tf.reduce_mean(mse)

        loss = mse_mean + regularization
        
        mae = tf.abs(average_O - Y)
        
        accuracy = tf.reduce_mean(mae)
        
    with tf.name_scope("Train") as scope:
        
        vars_A = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attention_module')
        vars_L = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='learning_module')
        vars_R = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Loss')
        
        # Minimizing Loss Function
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            loss, global_step=global_step,var_list=[vars_A,vars_L,vars_R])

    with tf.control_dependencies([optimizer, variable_averages_op]):
        train_op = tf.no_op(name='train')
        
    for op in [train_op, A]:
        tf.add_to_collection('train_ops', op)
        
    for op in [loss, accuracy]:
        tf.add_to_collection('validate_ops', op)

        
        
def test(input_size,train_X, train_Y, test_X, test_Y,total_batch,index):
    
    X = tf.placeholder(tf.float32, [None, input_size])
    Y = tf.placeholder(tf.float32, [None, output_size])
    global_step = tf.Variable(0, trainable=False)


    with tf.variable_scope("test_model_{}".format(index)) as scope:

        
        L_W1 = tf.Variable(
            tf.truncated_normal([input_size, L_node], stddev=0.1, seed=set_seed))
        
        L_b1 = tf.Variable(tf.constant(0.1, shape=[L_node]))
        
        L_W2 = tf.Variable(
            tf.truncated_normal([L_node, output_size], stddev=0.1, seed=set_seed))
        
        L_b2 = tf.Variable(tf.constant(0.1, shape=[output_size]))

        variable_averages = tf.train.ExponentialMovingAverage(
                moving_average_decay, global_step)
        
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables(scope='test_model_{}'.format(index)))
        
        L_FC = tf.nn.relu(tf.matmul(X, L_W1) + L_b1)
        
        O = tf.matmul(L_FC, L_W2) + L_b2

        average_L_FC = tf.nn.relu(tf.matmul(X, variable_averages.average(L_W1)) + variable_averages.average(L_b1))
        
        average_O = tf.matmul(average_L_FC, variable_averages.average(L_W2)) + variable_averages.average(L_b2)
        

        
    with tf.name_scope("test_Loss_{}".format(index)) as scope:
        
        regularizer = tfnew.regularizers.l2(
                regularization_rate)
        
        regularization = regularizer(L_W1) + regularizer(L_W2)

        learning_rate = tf.train.exponential_decay(
                learning_rate_base, global_step, total_batch,
                learning_rate_decay)

        mse = tf.square(average_O - Y)
        
        mse_mean = tf.reduce_mean(mse)

        loss = mse_mean + regularization

        mae = tf.abs(average_O - Y)
        
        accuracy = tf.reduce_mean(mae)
        
    with tf.name_scope("Train") as scope:
        
        vars_m = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='test_model_{}'.format(index))
        vars_l = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='test_Loss_{}'.format(index))
        
        # Minimizing Loss Function
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            loss, global_step=global_step, var_list=[vars_m, vars_l])


    with tf.control_dependencies([optimizer, variable_averages_op]):
        train_op = tf.no_op(name='train_{}'.format(index))

    Iterator = BatchCreate(train_X, train_Y)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for step in range(1, train_step + 1):
            
            xs, ys = Iterator.next_batch(batch_size)
            
            sess.run(train_op, feed_dict={X: xs, Y: ys})
            
        accuracy = sess.run(accuracy,
                            feed_dict={X: test_X, Y: test_Y})
    return accuracy



#Main

def run_train(sess, train_X, train_Y, val_X, val_Y):
    X = tf.get_collection('input')[0]
    Y = tf.get_collection('output')[0]
    
    Iterator = BatchCreate(train_X, train_Y)
    
    for step in range(1, train_step+1):
        if step % 1 == 0:
            val_loss,val_accuracy = sess.run(tf.get_collection('validate_ops'), 
                                                        feed_dict={X:val_X, Y:val_Y})
            
            print('[%4d] AFS-loss:%.4f AFS-MAE:%.4f'%
                        (step, val_loss,val_accuracy))
            
        xs, ys = Iterator.next_batch(batch_size)
        
        _, A = sess.run(tf.get_collection('train_ops'), feed_dict={X:xs, Y:ys})
                        
    return A

def run_test(A,train_X, train_Y,test_X, test_Y,total_batch):

    attention_weight = A.mean(0)
    AFS_wight_rank = list(np.argsort(attention_weight))[::-1]
    ac_score_list = []
    index=1
    
    for N in range(5, 105, 5):
        
        use_train_x = train_X[:, AFS_wight_rank[:N]]
        use_test_x = test_X[:, AFS_wight_rank[:N]]
        
        accuracy = test(N, use_train_x, train_Y, use_test_x, test_Y,total_batch,index)
        index += 1
        
        print('Using Top {} features| MAE:{:.4f}'.format(N,accuracy))

        ac_score_list.append(accuracy)
        
    return ac_score_list

def main(argv=None):
    
    data = pd.read_excel("file:///Users/sparshg/Desktop/DNAMethyl/Project/Attention/Final_Dataset/Healthy.xlsx")

    X = data.iloc[:, 1:101].values
    Y = data.iloc[:, 0].values


    train_X, test_X, train_Y, test_Y = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=9)

    test_X, val_X, test_Y, val_Y = train_test_split(test_X, test_Y,
                                                    test_size=0.3,
                                                    random_state=23)

    
    train_Y = train_Y.reshape(-1, 1)
    test_Y = test_Y.reshape(-1, 1)
    val_Y = val_Y.reshape(-1, 1)

    
    Train_size = len(train_X)
    total_batch = Train_size / batch_size
    
    build(total_batch)
    
    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()
        print('== Get feature weight by using AFS ==')
        A = run_train(sess, train_X, train_Y, val_X, val_Y)
        
    print('==  The Evaluation of AFS ==')
    ac_score_list = run_test(A, train_X, train_Y,test_X, test_Y,total_batch)
    
    show_result(ac_score_list, dataset_name)
    
if __name__ == '__main__':
    tf.app.run()
