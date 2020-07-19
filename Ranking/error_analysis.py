import pandas as pd
import pickle
import numpy as np
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing as prep
from itertools import permutations, combinations

with open('6._Ranking/datasets/getdisciplined_with_ranks', 'rb') as d:
    dataa = pickle.load(d)
vectors = dataa['Vector'].values
with open('6._Ranking/datasets/Advice_with_ranks', 'rb') as dd:
    datab = pickle.load(dd)
vectors2 = datab['Vector'].values
comms1 = dataa['Comment'].values
comms2 = datab['Comment'].values
komenty = np.concatenate((comms1, comms2), axis=0).reshape([-1, 1])
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
prep.normalize(newdata[:, :100], norm='l2', copy=False)
prep.normalize(newdata[:, 100:114], norm='l2', copy=False)
newdata = newdata[:, :114]
print(newdata.shape)

i = 1
for i in range(1, 11):
    #this part prints confusion matrices; see the Excel file for mean calculations of all 10 matrices
    print('Model', i)
    X_test = np.load('6._Ranking/models/X_test_'+str(i)+'.npy')
    Y_test = np.load('6._Ranking/models/Y_test_'+str(i)+'.npy')
    preds = np.load('6._Ranking/models/preds_'+str(i)+'.npy')
    true = np.load('6._Ranking/models/true_'+str(i)+'.npy')
    preds = softmax(preds, axis=-1)
    preds = np.argmax(preds, axis=-1)
    minus = preds-true
    inds = np.nonzero(minus)
    accuracy = 100-(len(inds[0])*100/len(true))
    print('Accuracy on the test set =', accuracy)
    print('Misranked comments:', len(inds[0]))
    conf = confusion_matrix(true, preds)
    print('Confusion matrix:')
    print(conf)
    #this part lets you see the misclassified comments
    print('Model', i, 'Misranked comments:', len(inds[0]))
    indexy = []
    for number in inds[0]:
        n = number
        onepred = preds[n]
        onetrue = true[n]
        print('True label:', onetrue, 'Predicted label:', onepred)
        i = n % 3
        full = int(np.floor(n/3))
        if i == 0:
            viktor = np.squeeze(X_test[full])
            vec = viktor[:114]
        if i == 1:
            viktor = np.squeeze(X_test[full])
            vec = viktor[114:228]
        if i == 2:
            viktor = np.squeeze(X_test[full])
            vec = viktor[228:]
        for ind, line in enumerate(newdata):
            if str(vec[0])[:-1] in str(line[0]):
                indexy.append(ind)
                print(ind, komenty[ind])