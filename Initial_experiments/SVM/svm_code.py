from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

#load datasets as needed
motiv_train_x = np.load('5.3._Initial_experiments/datasets/getdisciplined_training_set_X.npy')
motiv_test_x = np.load('5.3._Initial_experiments/datasets/getdisciplined_test_set_X.npy')
til_train_x = np.load('5.3._Initial_experiments/datasets/til_train_X.npy')
til_test_x = np.load('5.3._Initial_experiments/datasets/til_test_X.npy')
reladv_train_x = np.load('5.3._Initial_experiments/datasets/reladv_training_set_X.npy')
reladv_test_x = np.load('5.3._Initial_experiments/datasets/reladv_test_set_X.npy')
pics_train_x = np.load('5.3._Initial_experiments/datasets/pics_train_X.npy')
pics_test_x = np.load('5.3._Initial_experiments/datasets/pics_test_X.npy')
# advice_train_x = np.concatenate((motiv_train_x, reladv_test_x), axis=0)
# advice_test_x = np.concatenate((motiv_test_x, reladv_test_x), axis=0)
# nonadvice_train_x = np.concatenate((til_train_x, pics_train_x), axis=0)
# nonadvice_test_x = np.concatenate((til_test_x, pics_test_x), axis=0)
advice_train_x = motiv_train_x
advice_test_x = motiv_test_x
nonadvice_train_x = pics_train_x
nonadvice_test_x = pics_test_x

advice_train_y = np.ones([1, advice_train_x.shape[0]])
advice_test_y = np.ones([1, advice_test_x.shape[0]])
nonadvice_train_y = np.zeros([1, nonadvice_train_x.shape[0]])
nonadvice_test_y = np.zeros([1, nonadvice_test_x.shape[0]])
train_set_x = np.concatenate((advice_train_x, nonadvice_train_x), axis=0)
test_set_x = np.concatenate((advice_test_x, nonadvice_test_x), axis=0)
train_set_y = np.concatenate((advice_train_y, nonadvice_train_y), axis=1)
test_set_y = np.concatenate((advice_test_y, nonadvice_test_y), axis=1)
print('Train set examples:', train_set_x.shape[0])
print('Test set examples:', test_set_x.shape[0])

X_train = train_set_x
Y_train = train_set_y.T.ravel()
X_test = test_set_x
Y_test = test_set_y.T.ravel()

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

data = []
clf = svm.SVC(C=20, kernel='rbf')
clf.fit(X_train, Y_train)
y_train_hat = clf.predict(X_train)
y_test_hat = clf.predict(X_test)
train_accuracy = accuracy_score(Y_train, y_train_hat)
test_acccuracy = accuracy_score(Y_test, y_test_hat)
labels = ['Train_accuracy', 'Test_accuracy']
data.append((train_accuracy, test_acccuracy))
df = pd.DataFrame.from_records(data, columns=labels)
print(df.to_string(), file=open('svm_results.txt', 'w'))
