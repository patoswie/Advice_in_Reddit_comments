import numpy as np
import matplotlib as plt
import sklearn.metrics
import sklearn.datasets
import pandas as pd
import pickle
from sklearn import preprocessing as prep
from sklearn.model_selection import KFold

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.rand(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

def forward_propagation(X_train, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    Z1 = np.dot(W1, X_train) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1/(1+np.exp(-Z2))
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

def compute_cost(A2, Y_train):
    m = Y_train.shape[1]
    logprobs = np.multiply(Y_train, np.log(A2))+np.multiply((1-Y_train), np.log(1-A2))
    cost = -np.sum(logprobs)/m
    cost = np.squeeze(cost)
    return cost

def backward_propagation(parameters, cache, X_train, Y_train):
    m = X_train.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    dZ2 = A2-Y_train
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X_train.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

def nn_model(X_train, Y_train, n_h, n_x, n_y, num_iterations, learning_rate, print_cost=True):
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X_train, parameters)
        cost = compute_cost(A2, Y_train)
        grads = backward_propagation(parameters, cache, X_train, Y_train)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print('Cost after iteration ' + str(i) +': ' + str(cost))
            costs.append(cost)
    return parameters, costs

def predict(parameters, X_test, Y_test):
    A2, cache = forward_propagation(X_test, parameters)
    y_hat = (A2 > 0.5)
    accuracy = float((np.dot(Y_test, y_hat.T) + np.dot(1-Y_test, 1-y_hat.T))/float(Y_test.size))
    precision, recall, Fscore, support = sklearn.metrics.precision_recall_fscore_support(Y_test, y_hat, beta=1.0, average='micro')
    return accuracy, precision, recall, Fscore, support

#training the network
motiv = pd.read_pickle('5.6._Proposed_method/datasets/training/getdisciplined')
til = pd.read_pickle('5.6._Proposed_method/datasets/training/todayilearned')
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
motiv_y = np.concatenate((new_motiv, np.ones([new_motiv.shape[0], 1])), axis=1)
til_y = np.concatenate((new_til, np.zeros([new_til.shape[0], 1])), axis=1)
dataset = np.concatenate((motiv_y, til_y), axis=0)
number = dataset.shape[1] - 1
prep.normalize(dataset[:, :100], norm='l2', copy=False)

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

    print('Train set examples:', X_train.shape[1])
    print('Test set examples:', X_test.shape[1])

    n_x = X_train.shape[0]
    n_h = 10
    n_y = Y_train.shape[0]
    num_iterations = 20000
    learning_rate = 0.2

    params, costs = nn_model(X_train, Y_train, n_h, n_x, n_y, num_iterations, learning_rate, print_cost=True)
    with open('parameters_' + str(q) + '.pkl', 'wb') as f:
        pickle.dump(params, f)
    train_accuracy, train_precision, train_recall, train_Fscore, train_support = predict(params, X_train, Y_train)
    test_acccuracy, test_precision, test_recall, test_Fscore, test_support = predict(params, X_test, Y_test)
    data.append((train_accuracy, train_precision, train_recall, train_Fscore, train_support, test_acccuracy, test_precision, test_recall, test_Fscore, test_support))
    q = q+1
labels = ['Train_accuracy', 'Train_precision', 'Train_recall', 'Train_Fscore', 'Train_support', 'Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
df = pd.DataFrame.from_records(data, columns=labels)
print(df.to_string(), file=open('model_results.txt', 'w'))