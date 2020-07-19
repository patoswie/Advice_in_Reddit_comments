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
orig_dataset_with = np.concatenate((motiv_y2, til_y2), axis=0)
np.random.shuffle(orig_dataset_with)
prep.normalize(orig_dataset_with[:, :100], norm='l2', copy=False)
number_with = orig_dataset_with.shape[1] - 1
orig_shal_with_y = orig_dataset_with[:, number_with].real.astype(np.float32).reshape([1, -1])


#Embeddings-Advice order
exp_panda = []
for i in range(10):
    dataset_with = orig_dataset_with
    shal_with_y = orig_shal_with_y
    advice_params = pickle.load(open('5.6._Proposed_method/Proposed_method/Embeddings_Advice_order/Advice_Networks/parameters_'+str(i)+'.pkl', 'rb'))
    embeddings_params = pickle.load(open('5.6._Proposed_method/Proposed_method/Embeddings_Advice_order/Embeddings_parameters.pkl', 'rb'))
    A2_for_exp = get_a2_shal(learned=embeddings_params, data=dataset_with[:, :100].T.real.astype(np.float32))
    preds = np.round(A2_for_exp)
    embeddings_accuracy = float((np.dot(shal_with_y, A2_for_exp.T) + np.dot(1 - shal_with_y, 1 - A2_for_exp.T)) / float(shal_with_y.size))
    embeddings_precision, embeddings_recall, embeddings_Fscore, embeddings_support = sklearn.metrics.precision_recall_fscore_support(
        shal_with_y, preds, beta=1.0, average='micro')
    print('Embeddings results on the new test set', embeddings_accuracy, embeddings_precision, embeddings_recall, embeddings_Fscore)
    preds = preds.squeeze()
    to_mask = []
    count = 0
    for x in range(len(preds)):
        if (int(preds[x]) == 0):
            to_mask.append(x)
        if (int(preds[x])==0) and (int(shal_with_y[0][x])==1):
            count = count+1
    dataset_with = np.delete(dataset_with, to_mask, 0)
    shal_with_y = np.delete(shal_with_y, to_mask, 1)
    print('Relevant examples thrown away, first deletion:', count)
    print('New data shape:', dataset_with.shape)
    exp_a2 = get_a2_shal(advice_params, dataset_with[:, 100:114].T.real.astype(np.float32))
    test_precision, test_recall, test_Fscore, test_support = sklearn.metrics.precision_recall_fscore_support(shal_with_y, np.round(exp_a2), beta=1.0, average='micro')
    test_accuracy = float((np.dot(shal_with_y, exp_a2.T) + np.dot(1-shal_with_y, 1-exp_a2.T))/float(shal_with_y.size))
    line = [test_accuracy, test_precision, test_recall, test_Fscore, test_support]
    exp_panda.append(line)
    thrownaway_results = get_a2_shal(learned=embeddings_params, data=dataset_with[:, :100].T.real.astype(np.float32))
    embeddings_accuracy = float(
        (np.dot(shal_with_y, thrownaway_results.T) + np.dot(1 - shal_with_y, 1 - thrownaway_results.T)) / float(shal_with_y.size))
    embeddings_precision, embeddings_recall, embeddings_Fscore, embeddings_support = sklearn.metrics.precision_recall_fscore_support(
        shal_with_y, np.round(thrownaway_results), beta=1.0, average='micro')
    print('Embeddings results after throwing away', embeddings_accuracy, embeddings_precision, embeddings_recall, embeddings_Fscore)
    cnt = 0
    fx = np.round(exp_a2).squeeze()
    for x in range(len(fx)):
        if (int(fx[x]) == 0) and (int(shal_with_y[0][x]) == 1):
            cnt = cnt + 1
    print('Relevant examples thrown away, second deletion:', cnt)
labels = ['Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
df = pd.DataFrame.from_records(exp_panda, columns=labels)
mean = df.mean()
print('Mean results for Advice Networks:', mean)
print(df.to_string(), file=open('5.6._Proposed_method/Proposed_method/Embeddings_Advice_order/new_method_test_results.txt', 'w'))

#Advice-Embeddings order
exp_panda = []
for i in range(10):
    dataset_with = orig_dataset_with
    shal_with_y = orig_shal_with_y
    advice_params = pickle.load(open('5.6._Proposed_method/Proposed_method/Advice_Embeddings_order/Advice_parameters.pkl', 'rb'))
    embeddings_params = pickle.load(open('5.6._Proposed_method/Proposed_method/Advice_Embeddings_order/Embeddings_Networks/parameters_'+str(i)+'.pkl', 'rb'))
    A2_for_exp = get_a2_shal(learned=advice_params, data=dataset_with[:, 100:114].T.real.astype(np.float32))
    preds = np.round(A2_for_exp)
    advice_accuracy = float((np.dot(shal_with_y, A2_for_exp.T) + np.dot(1 - shal_with_y, 1 - A2_for_exp.T)) / float(shal_with_y.size))
    advice_precision, advice_recall, advice_Fscore, advice_support = sklearn.metrics.precision_recall_fscore_support(
        shal_with_y, preds, beta=1.0, average='micro')
    print('Advice results on the new test set', advice_accuracy, advice_precision, advice_recall, advice_Fscore)
    preds = preds.squeeze()
    to_mask = []
    count = 0
    for x in range(len(preds)):
        if (int(preds[x]) == 0):
            to_mask.append(x)
        if (int(preds[x])==0) and (int(shal_with_y[0][x])==1):
            count = count+1
    dataset_with = np.delete(dataset_with, to_mask, 0)
    shal_with_y = np.delete(shal_with_y, to_mask, 1)
    print('Relevant examples thrown away, first deletion:', count)
    print('New data shape:', dataset_with.shape)
    exp_a2 = get_a2_shal(embeddings_params, dataset_with[:, :100].T.real.astype(np.float32))
    test_precision, test_recall, test_Fscore, test_support = sklearn.metrics.precision_recall_fscore_support(shal_with_y, np.round(exp_a2), beta=1.0, average='micro')
    test_accuracy = float((np.dot(shal_with_y, exp_a2.T) + np.dot(1-shal_with_y, 1-exp_a2.T))/float(shal_with_y.size))
    line = [test_accuracy, test_precision, test_recall, test_Fscore, test_support]
    exp_panda.append(line)
    thrownaway_results = get_a2_shal(learned=advice_params, data=dataset_with[:, 100:114].T.real.astype(np.float32))
    advice_accuracy = float(
        (np.dot(shal_with_y, thrownaway_results.T) + np.dot(1 - shal_with_y, 1 - thrownaway_results.T)) / float(shal_with_y.size))
    advice_precision, advice_recall, advice_Fscore, advice_support = sklearn.metrics.precision_recall_fscore_support(
        shal_with_y, np.round(thrownaway_results), beta=1.0, average='micro')
    print('Advice results after throwing away', advice_accuracy, advice_precision, advice_recall, advice_Fscore)
    cnt = 0
    fx = np.round(exp_a2).squeeze()
    for x in range(len(fx)):
        if (int(fx[x])==0) and (int(shal_with_y[0][x])==1):
            cnt = cnt+1
    print('Relevant examples thrown away, second deletion:', cnt)
labels = ['Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore', 'Test_support']
df = pd.DataFrame.from_records(exp_panda, columns=labels)
mean = df.mean()
print('Mean results for Embeddings Networks:', mean)
print(df.to_string(), file=open('5.6._Proposed_method/Proposed_method/Advice_Embeddings_order/new_method_test_results_reversed.txt', 'w'))