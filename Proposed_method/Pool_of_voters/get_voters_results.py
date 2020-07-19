import numpy as np
from scipy import stats
import pandas as pd
import sklearn.metrics

#the 14 features models
voters_matrix = pd.read_pickle('5.6._Proposed_method/Pool_of_voters/14_voter_matrix')
true_labels = pd.read_pickle('5.6._Proposed_method/Pool_of_voters/true_labels')
best_nine = [0, 6, 9, 1, 7, 2, 8, 4, 3]
best_five = [0, 6, 9, 1, 7]
best_three = [0, 6, 9]
best_one = [0]
data = []
best_list = [best_nine, best_five, best_three, best_one]
for item in (best_list):
    masked_frame = voters_matrix[item, :]
    votes = stats.mode(masked_frame, axis=0)[0]
    accuracy = float((np.dot(true_labels, votes.T) + np.dot(1-true_labels, 1-votes.T))/float(true_labels.size))
    precision, recall, Fscore, support = sklearn.metrics.precision_recall_fscore_support(
        true_labels, votes, beta=1.0, average='micro')
    line = ['Best_'+str(len(item)), accuracy, precision, recall, Fscore]
    data.append(line)
labels = ['Models', 'Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore']
df = pd.DataFrame.from_records(data, columns=labels)
print(df.to_string(), file=open('14_voters_results.txt', 'w'))

#the 100 features models
voters_matrix = pd.read_pickle('5.6._Proposed_method/Pool_of_voters/100_voter_matrix')
true_labels = pd.read_pickle('5.6._Proposed_method/Pool_of_voters/true_labels')
best_nine = [3, 7, 1, 5, 2, 0, 6, 8, 4]
best_five = [3, 7, 1, 5, 2]
best_three = [3, 7, 1]
best_one = [3]
data = []
best_list = [best_nine, best_five, best_three, best_one]
for item in (best_list):
    masked_frame = voters_matrix[item, :]
    votes = stats.mode(masked_frame, axis=0)[0]
    accuracy = float((np.dot(true_labels, votes.T) + np.dot(1-true_labels, 1-votes.T))/float(true_labels.size))
    precision, recall, Fscore, support = sklearn.metrics.precision_recall_fscore_support(
        true_labels, votes, beta=1.0, average='micro')
    line = ['Best_'+str(len(item)), accuracy, precision, recall, Fscore]
    data.append(line)
labels = ['Models', 'Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore']
df = pd.DataFrame.from_records(data, columns=labels)
print(df.to_string(), file=open('100_voters_results.txt', 'w'))

#the 114 features models
voters_matrix = pd.read_pickle('5.6._Proposed_method/Pool_of_voters/114_voter_matrix')
true_labels = pd.read_pickle('5.6._Proposed_method/Pool_of_voters/true_labels')
best_nine = [8, 6, 0, 4, 5, 9, 1, 7, 2]
best_five = [8, 6, 0, 4, 5]
best_three = [8, 6, 0]
best_one = [8]
data = []
best_list = [best_nine, best_five, best_three, best_one]
for item in (best_list):
    masked_frame = voters_matrix[item, :]
    votes = stats.mode(masked_frame, axis=0)[0]
    accuracy = float((np.dot(true_labels, votes.T) + np.dot(1-true_labels, 1-votes.T))/float(true_labels.size))
    precision, recall, Fscore, support = sklearn.metrics.precision_recall_fscore_support(
        true_labels, votes, beta=1.0, average='micro')
    line = ['Best_'+str(len(item)), accuracy, precision, recall, Fscore]
    data.append(line)
labels = ['Models', 'Test_accuracy', 'Test_precision', 'Test_recall', 'Test_Fscore']
df = pd.DataFrame.from_records(data, columns=labels)
print(df.to_string(), file=open('114_voters_results.txt', 'w'))