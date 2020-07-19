import pandas as pd
import pickle
import numpy as np
import scipy.stats as stats

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
newdata = np.hstack((vectorsy[:, 100:], ranks))
newdata = np.split(newdata, 3690, axis=0)
jedynki = []
dwojki = []
trojki = []
for item in newdata:
    first = item[0]
    jedynki.append(first)
    scnd = item[1]
    dwojki.append(scnd)
    thrd = item[2]
    trojki.append(thrd)
jedynki = np.array(jedynki)
dwojki = np.array(dwojki)
trojki = np.array(trojki)
labels = ['aptitude', 'attention', 'pleasantness', 'sensitivity', 'relate', 'imperative', 'advice', 'ASD', 'ASH', 'ASHD', 'TOC', 'CNE', 'CPN', 'LEN', 'rank']
jed_mean = np.mean(jedynki, axis=0)
jed_median = np.median(jedynki, axis=0)
dw_mean = np.mean(dwojki, axis=0)
dw_median = np.median(dwojki, axis=0)
tr_mean = np.mean(trojki, axis=0)
tr_median = np.median(trojki, axis=0)
data = [jed_mean, jed_median, dw_mean, dw_median, tr_mean, tr_median]
df = pd.DataFrame.from_records(data, columns=labels)
print(df.to_string(), file=open('6._Ranking/ranks_stats.txt', 'w'))
data_p = []
for i in range(14):
    list1 = jedynki[:, i]
    list2 = dwojki[:, i]
    list3 = trojki[:, i]
    p1 = stats.ttest_ind(list1, list2)
    p2 = stats.ttest_ind(list2, list3)
    p3 = stats.ttest_ind(list1, list3)
    data_p.append([p1[1], p2[1], p3[1]])
labels_p = ['Ranks 0-1', 'Ranks 1-2', 'Ranks 0-2']
df1 = pd.DataFrame.from_records(data_p, columns = labels_p)
print(df1.to_string(), file=open('6._Ranking/p_values.txt', 'w'))