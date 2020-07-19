import csv
from scipy.stats.stats import pearsonr as pearson
import numpy as np
import collections

norm_score_list = []
cos_sim_word_list = []
cos_sim_doc_list = []
sent_dif_list = []
pol_dif_list = []
post_pol_list = []
com_pol_list = []
post_sent_list = []
com_sent_list = []
general_score_list = []

with open('5.1_Preliminary_studies/dataset.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        if row['Cosine_word2vec'] != '':
            x1 = row['Cosine_word2vec'].replace('[', '')
            x2 = x1.replace(']', '')
            cos_sim_word_list.append(float(x2))
        if row['Cosine_doc2vec'] != '':
            y1 = row['Cosine_doc2vec'].replace('[', '')
            y2 = y1.replace(']', '')
            cos_sim_doc_list.append(float(y2))
        sent_dif_list.append(float(row['Sent_dif']))
        pol_dif_list.append(float(row['Pol_dif']))
        post_sent_list.append(float(row['Post_Sent_pos']))
        com_sent_list.append((float(row['Com_Sent_pos'])))
        post_pol_list.append(float(row['Post_Polarity']))
        com_pol_list.append((float(row['Com_polarity'])))
        general_score_list.append(int(row['Com_score']))
        norm_score_list.append(float(row['Norm_score']))


# enter any two lists
print(pearson(post_pol_list, com_pol_list))
print(np.mean(cos_sim_word_list))
