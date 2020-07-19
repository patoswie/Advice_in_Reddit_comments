from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
from sklearn import preprocessing as prep, metrics
import random
from itertools import permutations, combinations
import math
import scipy.stats, scipy.special
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 114, kernel_size=(1, 100), stride=100)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=(1, 3), stride=3)
        self.fc1 = nn.Linear(38, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = x.view(-1, 3, 1, 114)
        x = torch.tanh(self.conv2(x))
        x = x.view(-1, 3, 38)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 3)
        return x

# <editor-fold desc="Minibatch creation">
def random_minibatches(X_train, Y_train, minibatch_size, seed):
    m = X_train.shape[0]
    np.random.seed(seed)
    minibatches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X_train[permutation, :, :]
    shuffled_Y = Y_train[permutation, :]
    num_complete_minibatches = math.floor(m/minibatch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[minibatch_size * k:minibatch_size * (k+1), :, :]
        mini_batch_Y = shuffled_Y[minibatch_size * k:minibatch_size * (k + 1), :]
        mini_batch_Y_flat = mini_batch_Y.flatten()
        minibatch = (mini_batch_X, mini_batch_Y_flat)
        minibatches.append(minibatch)
    if m % minibatch_size != 0:
        mini_batch_X = shuffled_X[minibatch_size * (math.floor(m/minibatch_size)):m, :, :]
        mini_batch_Y = shuffled_Y[minibatch_size * (math.floor(m / minibatch_size)):m, :]
        mini_batch_Y_flat = mini_batch_Y.flatten()
        minibatch = (mini_batch_X, mini_batch_Y_flat)
        minibatches.append(minibatch)
    return minibatches
# </editor-fold>

# <editor-fold desc="Plot figures">
def plot_results(X_test, Y_test):
    test_outputs = net(X_test)
    Y_test_flat = Y_test.flatten()
    test_loss = loss_function(test_outputs, Y_test_flat)
    values, predicted = torch.max(test_outputs, -1) #the argmax operation
    predicted = predicted.detach().numpy()
    ground_truth = Y_test.numpy()
    predicted = predicted.flatten()
    ground_truth = ground_truth.flatten()
    minus = ground_truth - predicted
    inds = np.nonzero(minus)
    accuracy = 100 - (len(inds[0]) * 100 / len(ground_truth))
    return test_loss, accuracy
# </editor-fold>

# <editor-fold desc="Data preprocessing">
np.set_printoptions(threshold=np.inf)
with open('6._Ranking/datasets/getdisciplined_with_ranks', 'rb') as d:
    dataa = pickle.load(d)
vectors = dataa['Vector'].values
with open('6._Ranking/datasets/Advice_with_ranks', 'rb') as dd:
    datab = pickle.load(dd)
vectors2 = datab['Vector'].values
new_vectors = np.zeros([vectors.shape[0], 114])
for x in range(len(vectors)):
    new_vectors[x][:] = vectors[x][0][0]
new_vectors2 = np.zeros([vectors2.shape[0], 114])
for x in range(len(vectors2)):
    new_vectors2[x][:] = vectors2[x][0][0]
vectorsy = np.concatenate((new_vectors, new_vectors2), axis=0)
ranks1 = dataa['Rank'].values.reshape([-1, 1])
ranks2 = datab['Rank'].values.reshape([-1, 1])
ranks = np.concatenate((ranks1, ranks2), axis=0)
newdata = np.hstack((vectorsy[:, :], ranks))
prep.normalize(newdata[:, 1:101], norm='l2', copy=False)
prep.normalize(newdata[:, 101:115], norm='l2', copy=False)

newdata = np.split(newdata, 3690, axis=0)
njudata = []
for item in newdata:
    first = item[0]
    scnd = item[1]
    thrd = item[2]
    itemlist = [first, scnd, thrd]
    perms = list(permutations(itemlist, r=3))
    allperms = []
    for item in perms:
        oneperm = np.stack(item, axis=0)
        njudata.append(oneperm)
newdata = np.stack(njudata, axis=0)
print(newdata.shape)
new1 = newdata[:, :, 1:101]
new2 = newdata[:, :, 115:]
newdata = np.dstack((new1, new2))
print(newdata.shape)

kfold = KFold(10, True, 1)
q = 1
data = []
for train, test in kfold.split(newdata):
    print('Now training model', q)
    print(newdata[train].shape, newdata[test].shape)

    # <editor-fold desc="Prepare data">
    X_train = newdata[train][:, :, :100].real.astype(np.float32)
    Y_train = np.subtract(newdata[train][:, :, 100].astype(np.float32), 1)
    X_test = newdata[test][:, :, :100].real.astype(np.float32)
    Y_test = np.subtract(newdata[test][:, :, 100].astype(np.float32), 1)
    X_train = np.expand_dims(np.expand_dims(X_train.reshape((X_train.shape[0], -1)), axis=1), axis=1)
    Y_train = Y_train.reshape((Y_train.shape[0], -1)).real.astype(np.float32)
    X_test = np.expand_dims(np.expand_dims(X_test.reshape((X_test.shape[0], -1)), axis=1), axis=1)
    Y_test = Y_test.reshape((Y_test.shape[0], -1)).real.astype(np.float32)
    X_train = torch.from_numpy(X_train)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_test = torch.from_numpy(X_test)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # </editor-fold>

    net = Net()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    minibatches = random_minibatches(X_train, Y_train, minibatch_size=512, seed=3)
    previous_loss = 0

    for epoch in range(1001):
        for minibatch in minibatches:
            (inputs, labels) = minibatch
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print('Training loss after epoch', epoch, loss.item())
            i = epoch / 100
            l, train_acc = plot_results(X_train, Y_train)
            print('Train accuracy:', train_acc)
            test_loss, test_acc = plot_results(X_test, Y_test)
            print('Test loss', test_loss.item())
            print('Test accuracy', test_acc)
    line = [q, loss.item(), train_acc, test_loss.item(), test_acc]
    data.append(line)
    q=q+1
labels = ['Fold', 'Training_loss', 'Training_accuracy', 'Test_loss', 'Test accuracy']
df = pd.DataFrame.from_records(data, columns=labels)
df.to_pickle('data_for_100_features')
print(df.to_string(), file = open('data_for_100_features.txt', 'w'))