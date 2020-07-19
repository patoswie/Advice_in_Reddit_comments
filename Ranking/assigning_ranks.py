import pickle
import numpy as np
import pandas as pd

def ranks(dataset_name):
    with open('6._Ranking/datasets/'+str(dataset_name), 'rb') as f:
        dataset = pickle.load(f)
    data = dataset.values
    ranks = [1, 2, 3]
    rankvec = np.tile(ranks, int(len(dataset)/3)).reshape([-1, 1])
    data = np.hstack((data, rankvec))
    return data

dataset_names = ['getdisciplined', 'Advice']
labels = ['Comment', 'Vector', 'Rank']
for name in dataset_names:
    frame = ranks(name)
    final_frame = pd.DataFrame.from_records(frame, columns=labels)
    with open('6._Ranking/datasets/'+str(name)+'_with_ranks', 'wb') as file:
        pickle.dump(final_frame, file)

#this creates the final dataset with 3 columns: 'Comment', 'Vector' and 'Rank'