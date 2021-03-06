import numpy as np
import matplotlib as plt
import sklearn.metrics
import sklearn.datasets
import pandas as pd
import pickle

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
#load the datasets as needed
motiv_train_x = np.load('5.3._Initial_experiments/datasets/getdisciplined_training_set_X.npy')
motiv_test_x = np.load('5.3._Initial_experiments/datasets/getdisciplined_test_set_X.npy')
til_train_x = np.load('5.3._Initial_experiments/datasets/til_train_X.npy')
til_test_x = np.load('5.3._Initial_experiments/datasets/til_test_X.npy')
reladv_train_x = np.load('5.3._Initial_experiments/datasets/reladv_training_set_X.npy')
reladv_test_x = np.load('5.3._Initial_experiments/datasets/reladv_test_set_X.npy')
pics_train_x = np.load('5.3._Initial_experiments/datasets/pics_train_X.npy')
pics_test_x = np.load('5.3._Initial_experiments/datasets/pics_test_X.npy')
# advice_train_x = np.concatenate((motiv_train_x, reladv_train_x), axis = 0)
# nonadvice_train_x = np.concatenate((til_train_x, pics_train_x), axis = 0)
# train_set_x = np.concatenate((advice_train_x, nonadvice_train_x), axis=0)
# advice_test_x = np.concatenate((motiv_test_x, reladv_test_x), axis = 0)
# nonadvice_test_x = np.concatenate((til_test_x, pics_test_x), axis = 0)
# test_set_x = np.concatenate((advice_test_x, nonadvice_test_x), axis=0)
advice_train_x = motiv_train_x
nonadvice_train_x = til_train_x
advice_test_x = motiv_test_x
nonadvice_test_x = til_test_x
train_set_x = np.concatenate((advice_train_x, nonadvice_train_x), axis=0)
test_set_x = np.concatenate((advice_test_x, nonadvice_test_x), axis=0)
advice_train_y = np.ones([1, advice_train_x.shape[0]])
advice_test_y = np.ones([1, advice_test_x.shape[0]])
nonadvice_train_y = np.zeros([1, nonadvice_train_x.shape[0]])
nonadvice_test_y = np.zeros([1, nonadvice_test_x.shape[0]])
train_set_y = np.concatenate((advice_train_y, nonadvice_train_y), axis=1)
test_set_y = np.concatenate((advice_test_y, nonadvice_test_y), axis=1)
print('Train set examples:', train_set_x.shape[0])
print('Test set examples:', test_set_x.shape[0])

X_train = train_set_x.T.real.astype(np.float64)
Y_train = train_set_y.real.astype(np.float64)
X_test = test_set_x.T.real.astype(np.float64)
Y_test = test_set_y.real.astype(np.float64)

n_x = X_train.shape[0]
n_h = 10
n_y = Y_train.shape[0]
num_iterations = 20000
learning_rate = 0.2

params, costs = nn_model(X_train, Y_train, n_h, n_x, n_y, num_iterations, learning_rate, print_cost=True)
with open('parameters.pkl', 'wb') as f:
    pickle.dump(params, f)
with open('costs.pkl', 'wb') as ff:
    pickle.dump(costs, ff)

#getting results
with open('parameters.pkl', 'rb') as f:
    parameters = pickle.load(f)
data = []
train_accuracy, train_precision, train_recall, train_Fscore, train_support = predict(parameters, X_train, Y_train)
test_acccuracy, test_precision, test_recall, test_Fscore, test_support = predict(parameters, X_test, Y_test)
labels = ['Train_accuracy', 'Train_precision', 'Train_recall', 'Train_Fscore', 'Train_support', 'Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
data.append((train_accuracy, train_precision, train_recall, train_Fscore, train_support, test_acccuracy, test_precision, test_recall, test_Fscore, test_support))
df = pd.DataFrame.from_records(data, columns=labels)
print(df.to_string(), file=open('model_results.txt', 'w'))
