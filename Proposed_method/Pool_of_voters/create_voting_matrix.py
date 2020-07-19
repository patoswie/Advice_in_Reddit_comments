import math
import numpy as np
import pickle
import pandas as pd
import sklearn.metrics
from sklearn import preprocessing as prep

def get_a2_shal(learned, data):
    Z1 = np.add(np.matmul(learned['W1'], data), learned['b1'])
    A1 = (np.exp(Z1)-np.exp(-Z1))/(np.exp(Z1)+np.exp(-Z1))
    Z2 = np.add(np.matmul(learned['W2'], A1), learned['b2'])
    A2 = 1/(1+np.exp(-Z2))
    return A2

motiv = pd.read_pickle('5.6._Proposed_method/datasets/test/new_test_getdisciplined')
til = pd.read_pickle('5.6._Proposed_method/datasets/test/new_test_todayilearned')
motiv1 = motiv.drop(columns = ['Comment']).values
til1 = til.drop(columns = ['Comment']).values
np.random.shuffle(motiv1)
np.random.shuffle(til1)
new_motiv = np.zeros([motiv1.shape[0], 114])
for x in range(len(motiv1)):
    new_motiv[x][:] = motiv1[x][0][0]
new_til = np.zeros([til1.shape[0], 114])
for x in range(len(til1)):
    new_til[x][:] = til1[x][0][0]
motiv_y2 = np.concatenate((new_motiv, np.ones([new_motiv.shape[0], 1])), axis=1)
til_y2 = np.concatenate((new_til, np.zeros([new_til.shape[0], 1])), axis=1)
dataset_with = np.concatenate((motiv_y2, til_y2), axis=0)
np.random.shuffle(dataset_with)
prep.normalize(dataset_with[:, :100], norm='l2', copy=False)
number_with = dataset_with.shape[1] - 1
shal_with_y = dataset_with[:, number_with].real.astype(np.float32).reshape([1, -1])

hundred_panda = []
hundred_matrix = []
for i in range(10):
    hundred_params = pickle.load(open('5.6._Proposed_method/Pool_of_voters/Models/100_features/parameters_'+str(i)+'.pkl', 'rb'))
    hundred_a2 = get_a2_shal(hundred_params, dataset_with[:, :100].T.real.astype(np.float32))
    hundred_result = np.round(hundred_a2).reshape([1, -1])
    hundred_matrix.append(hundred_result)
    test_precision, test_recall, test_Fscore, test_support = sklearn.metrics.precision_recall_fscore_support(shal_with_y, np.round(hundred_a2), beta=1.0, average='micro')
    test_accuracy = float((np.dot(shal_with_y, hundred_a2.T) + np.dot(1-shal_with_y, 1-hundred_a2.T))/float(shal_with_y.size))
    line = [test_accuracy, test_precision, test_recall, test_Fscore, test_support]
    hundred_panda.append(line)
labels = ['Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
df = pd.DataFrame.from_records(hundred_panda, columns=labels)
print(df.to_string(), file=open('100_new_test_results.txt', 'w'))
sto_matrix = np.squeeze(np.array(hundred_matrix))
with open('100_voter_matrix', 'wb') as m:
    pickle.dump(sto_matrix, m)

fourteen_panda = []
fourteen_matrix = []
for i in range(10):
    fourteen_params = pickle.load(open('5.6._Proposed_method/Pool_of_voters/Models/14_features/parameters_'+str(i)+'.pkl', 'rb'))
    fourteen_a2 = get_a2_shal(fourteen_params, dataset_with[:, 100:114].T.real.astype(np.float32))
    fourteen_result = np.round(fourteen_a2).reshape([1, -1])
    fourteen_matrix.append(fourteen_result)
    test_precision, test_recall, test_Fscore, test_support = sklearn.metrics.precision_recall_fscore_support(shal_with_y, np.round(fourteen_a2), beta=1.0, average='micro')
    test_accuracy = float((np.dot(shal_with_y, fourteen_a2.T) + np.dot(1-shal_with_y, 1-fourteen_a2.T))/float(shal_with_y.size))
    line = [test_accuracy, test_precision, test_recall, test_Fscore, test_support]
    fourteen_panda.append(line)
labels = ['Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
df = pd.DataFrame.from_records(fourteen_panda, columns=labels)
print(df.to_string(), file=open('14_new_test_results.txt', 'w'))
fourteen_matrix = np.squeeze(np.array(fourteen_matrix))
with open('14_voter_matrix', 'wb') as m:
    pickle.dump(fourteen_matrix, m)

all_panda = []
all_matrix = []
for i in range(10):
    all_params = pickle.load(open('5.6._Proposed_method/Pool_of_voters/Models/114_features/parameters_'+str(i)+'.pkl', 'rb'))
    all_a2 = get_a2_shal(all_params, dataset_with[:, :114].T.real.astype(np.float32))
    all_result = np.round(all_a2).reshape([1, -1])
    all_matrix.append(all_result)
    test_precision, test_recall, test_Fscore, test_support = sklearn.metrics.precision_recall_fscore_support(shal_with_y, np.round(all_a2), beta=1.0, average='micro')
    test_accuracy = float((np.dot(shal_with_y, all_a2.T) + np.dot(1-shal_with_y, 1-all_a2.T))/float(shal_with_y.size))
    line = [test_accuracy, test_precision, test_recall, test_Fscore, test_support]
    all_panda.append(line)
labels = ['Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
df = pd.DataFrame.from_records(all_panda, columns=labels)
print(df.to_string(), file=open('114_new_test_results.txt', 'w'))
all_matrix = np.squeeze(np.array(all_matrix))
with open('114_voter_matrix', 'wb') as m:
    pickle.dump(all_matrix, m)

#save the true labels vector
with open('true_labels', 'wb') as true:
    pickle.dump(shal_with_y, true)