import csv
import nltk, re, pprint
from nltk import word_tokenize
import codecs
import gensim, logging
from gensim import models
from gensim.models import doc2vec
from gensim.models import word2vec
import numpy as np
import scipy.spatial as sp
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import math as m
from collections import Counter
import numpy.linalg as ln
import pandas as pd

# analyze cosine similarity (doc2vec)
def doc2vec_analysis(line):
    post = line['Post_Text']
    comment = line['Com_text']
    post_tokenized = post
    com_tokenized = comment
    post_vector = model.infer_vector(post_tokenized)
    com_vector = model.infer_vector(com_tokenized)
    vec1 = np.array(post_vector).reshape(1, -1)
    vec2 = np.array(com_vector).reshape(1, -1)
    doc_cos_sim = 1 - sp.distance.cdist(vec1, vec2, 'cosine')
    return doc_cos_sim

# analyze cosine similarity (word2vec)
def wektor(text):
    vyrds = nltk.word_tokenize(text)
    sents = nltk.sent_tokenize(text)
    newsents = []
    sentcount = len(sents)
    a = m.pow(10, -3)
    sent_vec_list = []
    for x in range(sentcount):
        newsent = nltk.word_tokenize(sents[x])
        newsents.append(newsent)
    for newsent in newsents:
        sent_vec = 0
        uniq_words = Counter(vyrds)
        for i in range(len(newsent)):
            word = newsent[i]
            try:
                wordvec = model[word] * (a/(a+(uniq_words[word]/len(vyrds))))
            except KeyError:
                wordvec = 0
                iter = len(word)
                for x in range(iter):
                    cr = word[x]
                    wordvec = wordvec + model_char[cr]
                wordvec = wordvec/len(word)
            sent_vec = sent_vec + wordvec
        sent_vec = (1/len(newsent)) * sent_vec
        sent_vec_list.append(sent_vec)
    matrix_poz = np.matrix(sent_vec_list)
    matrix_pion = matrix_poz.transpose()
    matrix_B = np.matmul(matrix_pion, matrix_poz)
    w, v = ln.eig(matrix_B)
    for i, value in enumerate(w):
        if np.absolute(value-max(w))<1e-15:
            u_t = v[i]
    u = u_t.transpose()
    newsent_vec_list = []
    for index, newsent in enumerate(sent_vec_list):
        newsent_vec = sent_vec_list[index] - (np.matmul(np.matmul(u, u_t), sent_vec_list[index]))
        newsent_vec_list.append(newsent_vec)
    text_vec = 0
    for i in range(len(newsent_vec_list)):
        text_vec = text_vec + newsent_vec_list[i]
    return(text_vec)

def word2vec_analysis(line):
    post = ''.join([chr for chr in line['Post_Text'] if (32<=ord(chr) and ord(chr)<127)])
    comm = ''.join([chr for chr in line['Com_text'] if (32 <= ord(chr) and ord(chr) < 127)])
    post_vector = wektor(post)
    com_vector = wektor(comm)
    vec1 = np.array(post_vector).reshape(1, -1)
    vec2 = np.array(com_vector).reshape(1, -1)
    vec_cos_sim = 1 - sp.distance.cdist(vec1, vec2, 'cosine')
    return vec_cos_sim

# analyze polarity
def polarity_analysis(line):
    post_blob = TextBlob(line['Post_Text'])
    com_blob = TextBlob(line['Com_text'])
    com_polarity = com_blob.sentiment.polarity
    post_polarity = post_blob.sentiment.polarity
    return post_polarity, com_polarity

# analyze sentiment
def sentiment_analysis(line):
    post_sent = SentimentIntensityAnalyzer().polarity_scores(line['Post_Text'])
    com_sent = SentimentIntensityAnalyzer().polarity_scores(line['Com_text'])
    return post_sent['compound'], post_sent['pos'], post_sent['neu'], post_sent['neg'], com_sent['compound'], com_sent['pos'], com_sent['neu'], com_sent['neg']

def get_normscore(line):
    post_score = line['Post_Score']
    com_score = line['Com_score']
    norm_score = ((com_score*100)/post_score)/100
    return norm_score

doc2vec_model = doc2vec.Doc2Vec.load('doc2vec_model')
model = word2vec.Word2Vec.load('word2vec_model')
model_char = word2vec.Word2Vec.load('word2vec_model_chars')
fieldnames = ['Title', 'Id', 'Coms_no',	'Post_Text', 'Post_Score', 'Com_text', 'Com_score', 'Norm_score', 'Cosine_word2vec', 'Cosine_doc2vec', 'Post_Sent_comp', 'Post_Sent_pos', 'Post_Sent_neu', 'Post_Sent_neg',	'Com_Sent_comp', 'Com_Sent_pos', 'Com_Sent_neu', 'Com_Sent_neg', 'Sent_dif', 'Post_Polarity', 'Com_polarity', 'Pol_dif']
frame = pd.read_csv('5.1_Preliminary_studies/raw_dataset.csv', sep=';')
data = []
for i in range(6):
    title = frame.iloc[i]['Title']
    id = frame.iloc[i]['Id']
    coms_no = frame.iloc[i]['Coms_no']
    post_text = frame.iloc[i]['Post_Text']
    post_score = frame.iloc[i]['Post_Score']
    com_text = frame.iloc[i]['Com_text']
    com_score = frame.iloc[i]['Com_score']
    norm_score = get_normscore(frame.iloc[i])
    cosine_word2vec = word2vec_analysis(frame.iloc[i])
    cosine_doc2vec = doc2vec_analysis(frame.iloc[i])
    post_sent_comp, post_sent_pos, post_sent_neu, post_sent_neg, com_sent_comp, com_sent_pos, \
    com_sent_neu, com_sent_neg = sentiment_analysis(frame.iloc[i])
    sent_diff = (post_sent_pos - post_sent_neg) - (com_sent_pos - com_sent_neg)
    post_pol, com_pol = polarity_analysis(frame.iloc[i])
    pol_diff = post_pol-com_pol
    line = [title, id, coms_no, post_text, post_score, com_text, com_score, norm_score, cosine_word2vec, cosine_doc2vec, post_sent_comp, post_sent_pos, post_sent_neu, post_sent_neg, com_sent_comp, com_sent_pos, com_sent_neu, com_sent_neg, sent_diff, post_pol, com_pol, pol_diff]
    data.append(line)

frame = pd.DataFrame.from_records(data, columns=fieldnames)
frame.to_csv('5.1._Preliminary_studies/dataset.csv', encoding='utf8', sep=';', index=False)
