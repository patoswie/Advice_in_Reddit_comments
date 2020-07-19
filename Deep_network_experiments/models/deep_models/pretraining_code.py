import numpy as np
import matplotlib as plt
import sklearn.metrics
import sklearn.datasets
import pickle
import pandas as pd
import random
import math

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh(Z):
    A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    cache = Z
    return A, cache

def tanh_backward(dA, activation_cache):
    Z = activation_cache
    t = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    dZ = dA * (1-np.power(t, 2))
    return dZ

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

def initialize_parameters(layer_dims): #He for Relu or Xavier for tanh
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*np.sqrt(1/layer_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for i in range (L):
        v['dW' + str(i+1)] = np.zeros((parameters['W' + str(i+1)].shape[0], parameters['W'+str(i+1)].shape[1]))
        v['db' + str(i+1)] = np.zeros((parameters['b' + str(i+1)].shape[0], parameters['b' + str(i+1)].shape[1]))
        s['dW' + str(i + 1)] = np.zeros((parameters['W' + str(i + 1)].shape[0], parameters['W' + str(i + 1)].shape[1]))
        s['db' + str(i + 1)] = np.zeros((parameters['b' + str(i + 1)].shape[0], parameters['b' + str(i + 1)].shape[1]))
    return v, s

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_forward_activation(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    if activation == 'tanh':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_propagation(X_train, parameters):
    caches = []
    A = X_train
    L = len(parameters) // 2
    for i in range(1, L):
        A_prev = A
        A, cache = linear_forward_activation(A_prev, parameters['W'+str(i)], parameters['b'+str(i)], 'tanh')
        caches.append(cache)
    AL, cache = linear_forward_activation(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y_train):
    m = Y_train.shape[1]
    logprobs = np.multiply(Y_train, np.log(AL))+np.multiply((1-Y_train), np.log(1-AL))
    cost = -1/m*np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_backward_activation(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'tanh':
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_propagation(AL, Y_train, caches):
    grads = {}
    L = len(caches)
    Y_train = Y_train.reshape(AL.shape)
    dAL = -(np.divide(Y_train, AL)-np.divide(1-Y_train, 1-AL))
    current_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = linear_backward_activation(dAL, current_cache, 'sigmoid')
    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = linear_backward_activation(grads['dA'+str(i+1)], current_cache, 'tanh')
        grads['dA'+str(i)] = dA_prev_temp
        grads['dW'+str(i+1)] = dW_temp
        grads['db'+str(i+1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate, v, s, t, beta1=0.9, beta2 = 0.999, epsilon = 1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for i in range(L):
        v['dW'+str(i+1)] = beta1*v['dW'+str(i+1)]+(1-beta1)*grads['dW'+str(i+1)]
        v['db' + str(i+1)] = beta1 * v['db'+str(i+1)] + (1 - beta1) * grads['db' + str(i + 1)]
        v_corrected['dW'+str(i+1)] = v['dW'+str(i+1)]/(1-np.power(beta1, t+1))
        v_corrected['db' + str(i + 1)] = v['db' + str(i + 1)] / (1 - np.power(beta1, t+1))
        s['dW' + str(i + 1)] = beta2 * s['dW' + str(i + 1)] + (1 - beta2) * np.power(grads['dW' + str(i + 1)], 2)
        s['db' + str(i + 1)] = beta2 * s['db' + str(i + 1)] + (1 - beta2) * np.power(grads['db' + str(i + 1)], 2)
        s_corrected['dW' + str(i + 1)] = s['dW' + str(i + 1)] / (1 - np.power(beta2, t+1))
        s_corrected['db' + str(i + 1)] = s['db' + str(i + 1)] / (1 - np.power(beta2, t+1))
        parameters['W'+str(i+1)] = parameters['W'+str(i+1)] - learning_rate*(v_corrected['dW'+str(i+1)]/(np.sqrt(s_corrected['dW'+str(i+1)])+epsilon))
        parameters['b'+str(i+1)] = parameters['b'+str(i+1)] - learning_rate*(v_corrected['db'+str(i+1)]/(np.sqrt(s_corrected['db'+str(i+1)])+epsilon))
    return parameters

def deep_model(X_train, Y_train, layer_dims, learning_rate, iter, minibatch_size, beta1, print_cost=False):
    costs = []
    parameters = initialize_parameters(layer_dims)
    v, s = initialize_adam(parameters)
    seed = 0
    for t in range(0, iter):
        seed = seed + 1
        minibatches = random_minibatches(X_train, Y_train, minibatch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            AL, caches = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(AL, minibatch_Y)
            grads = backward_propagation(AL, minibatch_Y, caches)
            parameters = update_parameters(parameters, grads, learning_rate, v, s, t, beta1, beta2 = 0.999, epsilon = 1e-8)
        if print_cost and t % 1000 ==0:
            print('Cost after iteration ' + str(t) +': ' + str(cost))
        if print_cost and t % 1000 == 0:
            costs.append(cost)
    return parameters, costs

def predict(parameters, X_test, Y_test):
    AL, cache = forward_propagation(X_test, parameters)
    predictions = y_hat = (AL > 0.5)
    accuracy = float((np.dot(Y_test, y_hat.T) + np.dot(1-Y_test, 1-y_hat.T))/float(Y_test.size))
    differences = y_hat - Y_test
    precision, recall, Fscore, support = sklearn.metrics.precision_recall_fscore_support(Y_test, y_hat, beta=1.0, average = 'micro')
    return predictions, accuracy, differences, precision, recall, Fscore, support

def training(model_count):
    first = X_train.shape[0]
    last = 1
    learning_rate = 0.0037
    iter = 30000
    minibatch_size = X_train.shape[1]
    layer_dims = [7, 8, 15]
    layer_dims.append(last)
    layer_dims.insert(0, first)
    data = []
    for i in range(model_count):
        print('Now training model ', str(i))
        params, costs = deep_model(X_train, Y_train, layer_dims, learning_rate, iter, minibatch_size, beta1=0.9, print_cost=True)
        with open('5.4._Deep_network_experiments/models/deep_models/pretrained_model/parameters'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(params, f)
        with open('5.4._Deep_network_experiments/models/deep_models/pretrained_model/costs'+str(i)+'.pkl', 'wb') as ff:
            pickle.dump(costs, ff)
        pred, train_accuracy, differences, train_precision, train_recall, train_Fscore, support = predict(params, X_train, Y_train)
        pred, dev_acccuracy, differences, dev_precision, dev_recall, dev_Fscore, support = predict(params, X_dev, Y_dev)
        pred, test_acccuracy, differences, test_precision, test_recall, test_Fscore, support = predict(params, X_test, Y_test)
        labels = ['Train_accuracy', 'Train_precision', 'Train_recall', 'Train_Fscore', 'Dev_accuracy', 'Dev_precision', 'Dev_recall', 'Dev_Fscore', 'Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore']
        data.append((train_accuracy, train_precision, train_recall, train_Fscore, dev_acccuracy, dev_precision, dev_recall, dev_Fscore, test_acccuracy, test_precision, test_recall, test_Fscore))
    df = pd.DataFrame.from_records(data, columns=labels)
    df.to_pickle('5.4._Deep_network_experiments/models/deep_models/pretrained_model/modele/model_results')
    print(df.to_string(), file=open('5.4._Deep_network_experiments/models/deep_models/pretrained_model/model_results.txt', 'w'))

reladv_df = pd.read_pickle('/Users/patrycja/Desktop/doktorat/5.4._Deep_network_experiments/models/deep_models/datasets/data_for_pretraining/relationship_advice')
legaladv_df = pd.read_pickle('/Users/patrycja/Desktop/doktorat/5.4._Deep_network_experiments/models/deep_models/datasets/data_for_pretraining/legaladvice')
adv_df = pd.read_pickle('/Users/patrycja/Desktop/doktorat/5.4._Deep_network_experiments/models/deep_models/datasets/data_for_pretraining/Advice')
til_df = pd.read_pickle('/Users/patrycja/Desktop/doktorat/5.4._Deep_network_experiments/models/deep_models/datasets/data_for_pretraining/todayilearned')
pics_df = pd.read_pickle('/Users/patrycja/Desktop/doktorat/5.4._Deep_network_experiments/models/deep_models/datasets/data_for_pretraining/pics')
reladv_mx = reladv_df.drop(columns=['Comment']).values
legaladv_mx = legaladv_df.drop(columns=['Comment']).values
adv_mx = adv_df.drop(columns=['Comment']).values
til_mx = til_df.drop(columns = ['Comment']).values
pics_mx = pics_df.drop(columns = ['Comment']).values
np.random.shuffle(reladv_mx)
np.random.shuffle(legaladv_mx)
np.random.shuffle(adv_mx)
np.random.shuffle(til_mx)
np.random.shuffle(pics_mx)
advice_train_set = np.concatenate((np.concatenate((reladv_mx[:3588, :], legaladv_mx[:2612, :]), axis=0), adv_mx[:3396, :]), axis=0)
advice_train_set = np.concatenate((advice_train_set, np.ones([advice_train_set.shape[0], 1])), axis=1)
advice_dev_set = np.concatenate((np.concatenate((reladv_mx[3588:4036, :], legaladv_mx[2612:2938, :]), axis=0), adv_mx[3396:3820, :]), axis=0)
advice_dev_set = np.concatenate((advice_dev_set, np.ones([advice_dev_set.shape[0], 1])), axis=1)
advice_test_set = np.concatenate((np.concatenate((reladv_mx[4036:, :], legaladv_mx[2938:, :]), axis=0), adv_mx[3820:, :]), axis=0)
advice_test_set = np.concatenate((advice_test_set, np.ones([advice_test_set.shape[0], 1])), axis=1)
nonadvice_train_set = np.concatenate((til_mx[:3864, :], pics_mx[:2719, :]), axis=0)
nonadvice_train_set = np.concatenate((nonadvice_train_set, np.zeros([nonadvice_train_set.shape[0], 1])), axis=1)
nonadvice_dev_set = np.concatenate((til_mx[3864:4347, :], pics_mx[2719:3059, :]), axis=0)
nonadvice_dev_set = np.concatenate((nonadvice_dev_set, np.zeros([nonadvice_dev_set.shape[0], 1])), axis=1)
nonadvice_test_set = np.concatenate((til_mx[4347:, :], pics_mx[3059:, :]), axis=0)
nonadvice_test_set = np.concatenate((nonadvice_test_set, np.zeros([nonadvice_test_set.shape[0], 1])), axis=1)
train_set_x = np.concatenate((advice_train_set, nonadvice_train_set), axis=0)
dev_set_x = np.concatenate((advice_dev_set, nonadvice_dev_set), axis=0)
test_set_x = np.concatenate((advice_test_set, nonadvice_test_set), axis=0)
np.random.shuffle(train_set_x)
np.random.shuffle(dev_set_x)
np.random.shuffle(test_set_x)
number = train_set_x.shape[1] - 1
X_train = train_set_x[:, :number].T.real.astype(np.float64)
Y_train = train_set_x[:, number].T.real.astype(np.float64).reshape([1, -1])
X_dev = dev_set_x[:, :number].T.real.astype(np.float64)
Y_dev = dev_set_x[:, number].T.real.astype(np.float64).reshape([1, -1])
X_test = test_set_x[:, :number].T.real.astype(np.float64)
Y_test = test_set_x[:, number].T.real.astype(np.float64).reshape([1, -1])

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

training(model_count=10)
