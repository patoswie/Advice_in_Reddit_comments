import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import KFold
from sklearn import preprocessing as prep

motiv = pd.read_pickle('5.5._Word2vec_experiments/datasets/getdisciplined')
til = pd.read_pickle('5.5._Word2vec_experiments/datasets/todayilearned')
motiv1 = motiv.drop(columns = ['Comment']).values
til1 = til.drop(columns = ['Comment']).values
np.random.shuffle(motiv1)
np.random.shuffle(til1)
new_motiv = np.zeros([889, 114])
for x in range(len(motiv1)):
    new_motiv[x][:] = motiv1[x][0][0]
new_til = np.zeros([889, 114])
for x in range(len(til1)):
    new_til[x][:] = til1[x][0][0]
motiv_y = np.concatenate((new_motiv[:, :100], np.ones([new_motiv.shape[0], 1])), axis=1)
til_y = np.concatenate((new_til[:, :100], np.zeros([new_til.shape[0], 1])), axis=1)
dataset = np.concatenate((motiv_y, til_y), axis=0)
number = dataset.shape[1] - 1
prep.normalize(dataset[:, :100], norm='l2', copy=False)

def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=(n_x, None))
    Y = tf.placeholder(dtype=tf.float32, shape=(n_y, None))
    return X, Y

def initialize_parameters(L, layer_dims):
    parameters = {}
    for x in range(L + 1):
        parameters['W' + str(x + 1)] = tf.get_variable('W' + str(x + 1), [layer_dims[x + 1], layer_dims[x]],
                                                       initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(x + 1)] = tf.get_variable('b' + str(x + 1), [layer_dims[x + 1], 1],
                                                       initializer=tf.contrib.layers.xavier_initializer())
    return parameters

def forward_propagation(X, parameters, keep_prob):
    L = (len(parameters) // 2)-1
    Z = tf.add(tf.matmul(parameters['W1'], X), parameters['b1'])
    Z_dropped = tf.nn.dropout(Z, keep_prob=keep_prob)
    for x in range(1, L + 1):
        A = tf.nn.tanh(Z_dropped)
        Z = tf.add(tf.matmul(parameters['W' + str(x + 1)], A), parameters['b' + str(x + 1)])
        if x != L + 1:
            Z_dropped = tf.nn.dropout(Z, keep_prob=keep_prob)
        else:
            Z = Z
    return Z

def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

def random_minibatches(X_train, Y_train, minibatch_size, seed):
    np.random.seed(seed)
    m = X_train.shape[1]
    minibatches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X_train[:, permutation]
    shuffled_Y = Y_train[:, permutation].reshape((1, m))
    num_complete_minibatches = math.floor(m/minibatch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, minibatch_size * k:minibatch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:, minibatch_size * k:minibatch_size * (k + 1)]
        minibatch = (mini_batch_X, mini_batch_Y)
        minibatches.append(minibatch)
    if m % minibatch_size != 0:
        mini_batch_X = shuffled_X[:, minibatch_size * (math.floor(m/minibatch_size)):m]
        mini_batch_Y = shuffled_Y[:, minibatch_size * (math.floor(m / minibatch_size)):m]
        minibatch = (mini_batch_X, mini_batch_Y)
        minibatches.append(minibatch)
    return minibatches

kfold = KFold(10, True, 1)
q = 0
data = []
for train, test in kfold.split(dataset):
    print(q)
    X_train = dataset[train][:, :number].T.real.astype(np.float32)
    Y_train = dataset[train][:, number].real.astype(np.float32).reshape([1, -1])
    X_test = dataset[test][:, :number].T.real.astype(np.float32)
    Y_test = dataset[test][:, number].real.astype(np.float32).reshape([1, -1])
    print(X_train.shape, Y_train.shape)

    def model(q, X_train, Y_train, X_test, Y_test, alpha=0.0037, epochs=10000, minibatch_size=X_train.shape[1], keep_prob=1.0, print_cost=True):
        ops.reset_default_graph()
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        test_costs = []
        X, Y = create_placeholders(n_x, n_y)
        parameters = initialize_parameters(L=3, layer_dims=[100, 7, 8, 15, 1])
        Z = forward_propagation(X, parameters, keep_prob=keep_prob)
        cost = compute_cost(Z, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                epoch_cost = 0
                num_minibatches = int(m / minibatch_size)
                minibatches = random_minibatches(X_train, Y_train, minibatch_size, seed=1)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches
                if print_cost == True and epoch % 1000 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 1000 == 0:
                    costs.append(epoch_cost)
                    Z10_test = forward_propagation(X_test, parameters, keep_prob=1.0)
                    test_cost = compute_cost(Z10_test, Y_test).eval()
                    test_costs.append(test_cost)
            parameters = sess.run(parameters)
            print("Parameters have been trained!")
            preds = tf.nn.sigmoid(Z)  # here is the sigmoid
            correct_prediction = tf.equal(tf.round(preds), Y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            Z_train = forward_propagation(X_train, parameters, keep_prob=1.0)
            A_train = tf.nn.sigmoid(Z_train)
            train_precision, train_recall, train_Fscore, train_support = sklearn.metrics.precision_recall_fscore_support(Y_train, tf.round(A_train).eval(), beta=1.0, average='micro')
            Z_test = forward_propagation(X_test, parameters, keep_prob=1.0)
            A_test = tf.nn.sigmoid(Z_test)
            test_precision, test_recall, test_Fscore, test_support = sklearn.metrics.precision_recall_fscore_support(Y_test, tf.round(A_test).eval(), beta=1.0, average='micro')
            data.append((train_accuracy, train_precision, train_recall, train_Fscore, train_support, test_accuracy, test_precision, test_recall, test_Fscore, test_support))
            return parameters, costs, test_costs
    parameters, costs, test_costs = model(q, X_train, Y_train, X_test, Y_test)
    with open('parameters_' + str(q) + '.pkl', 'wb') as f:
        pickle.dump(parameters, f)
    with open('costs_' + str(q) + '.pkl', 'wb') as ff:
        pickle.dump(costs, ff)
    with open('test_costs_' + str(q) + '.pkl', 'wb') as fff:
        pickle.dump(test_costs, fff)
    q = q+1
labels = ['Train_accuracy', 'Train_precision', 'Train_recall', 'Train_Fscore', 'Train_support', 'Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
df = pd.DataFrame.from_records(data, columns=labels)
df.to_pickle('model_results')
print(df.to_string(), file=open('model_results.txt', 'w'))