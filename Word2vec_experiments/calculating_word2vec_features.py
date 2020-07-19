import nltk, re, pprint
from nltk import word_tokenize, sent_tokenize
import codecs
import gensim, logging
from gensim import models
from gensim.models import word2vec
import glob
import os
import sys
import numpy as np
import scipy.spatial as sp
import pandas as pd
import math as m
from collections import Counter
from symspellpy.symspellpy import SymSpell, Verbosity
import pickle

def expandContractions(text): #https://devpost.com/software/contraction-expander
    new = text.replace('’', '\'')
    split = new.split(' ')
    with open('5.5._Word2vec_experiments/tools_for_spellcheck/newdict.pkl', 'rb') as f:
        cdict = pickle.load(f)
    keylist = [key.lower().encode('utf8').decode('utf8') for key, value in cdict.items()]
    newlist = []
    for item in split:
        try:
            if item.lower() in keylist:
                newlist.append(cdict[item.lower().encode('utf8').decode('utf8')])
            else:
                newlist.append(item)
        except KeyError:
            print(item)
    tekisuto = ' '.join(newlist)
    return tekisuto

def spellcheck(text):
    max_dictionary_edit_distance = 2
    prefix_length = 7
    sym_spell = SymSpell(max_dictionary_edit_distance=max_dictionary_edit_distance, prefix_length=prefix_length)
    dictionary_path = '5.5._Word2vec_experiments/tools_for_spellcheck/frequency_dictionary_en_82_765.txt'
    term_index = 0
    count_index = 1
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return
    final_text = ''
    newtext = expandContractions(text)
    wordlist = nltk.word_tokenize(newtext.lower())
    for item in wordlist:
        if item in '.,:;?!-':
            final_text = final_text + item
        elif item == 'i':
            final_text = final_text + ' ' + item
        elif (item == 'ive'):
            final_text = final_text + ' i have'
        elif (item == 'id'):
            final_text = final_text + ' i would'
        elif (item == 'im'):
            final_text = final_text + ' i am'
        elif (item == 'dont'):
            final_text = final_text + ' do not'
        else:
            input_term = item
            max_edit_distance_lookup = 2
            suggestion_verbosity = Verbosity.TOP  # TOP, CLOSEST, ALL
            suggestions = sym_spell.lookup(input_term, suggestion_verbosity,
                                           max_edit_distance_lookup)
            if len(suggestions) == 0: #if suggestion not found, then leave as is to avoid deleting words
                final_text = final_text + ' ' + input_term
            else:
                for suggestion in suggestions:
                    final_text = final_text + ' ' + str(suggestion.term)
    return final_text

def wektor(text):
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
        uniq_words = Counter(newsent)
        for i in range(len(newsent)):
            word = newsent[i]
            try:
                wordvec = model[word] * (a/(a*(uniq_words[word]/len(newsent))))
            except KeyError as e:
                wordvec = 0
                iter = len(word)
                for x in range(iter):
                    cr = word[x]
                    wordvec = wordvec + model_char[cr]
                wordvec = wordvec/len(word)
            sent_vec = sent_vec + wordvec
        sent_vec = (1/sentcount) * sent_vec
        sent_vec_list.append(sent_vec)
    matrix_poz = np.matrix(sent_vec_list)
    matrix_pion = matrix_poz.transpose()
    matrix_B = np.matmul(matrix_pion, matrix_poz)
    w, v = np.linalg.eig(matrix_B)
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
    text_vec = np.divide(text_vec, len(newsent_vec_list))
    return(text_vec)

model = word2vec.Word2Vec.load('5.5._Word2vec_experiments/word2vec_models/word2vec_model')
model_char = word2vec.Word2Vec.load('5.5._Word2vec_experiments/word2vec_models/word2vec_model_chars')

#First calculate the 14 advice features using the calculating_advice_features.py code. The code creates pickles, which will be opened and used in this code.

motiv = pd.read_pickle('getdisciplined')
motiv_dropped = motiv.drop('Comment', axis=1).values
til = pd.read_pickle('todayilearned')[:889]
til_dropped = til.drop('Comment', axis=1).values
with_features = []
for i in range(len(motiv)):
    print('motiv', i)
    comtext = motiv['Comment'][i]
    cleaned = ''.join([chr for chr in comtext if (32 <= ord(chr) and ord(chr) < 127)])  # clean to only ascii
    checked = spellcheck(cleaned)
    vec = wektor(checked)
    rest = motiv_dropped[i][:].reshape([1,-1])
    big_vec = np.concatenate((vec, rest), axis=1)
    with_features.append((comtext, big_vec))
labels = ['Comment', 'Vector']
df = pd.DataFrame.from_records(with_features, columns=labels)
df.to_pickle('5.5._Word2vec_experiments/datasets/getdisciplined')
print(df.to_string(), file=open('5.5._Word2vec_experiments/datasets/getdisciplined.txt', 'w'))
with_features = []
for i in range(len(til)):
    print('til', i)
    comtext = til['Comment'][i]
    cleaned = ''.join([chr for chr in comtext if (32<=ord(chr) and ord(chr)<127)]) #clean to only ascii
    checked = spellcheck(cleaned)
    vec = wektor(checked)
    rest = til_dropped[i][:].reshape([1,-1])
    big_vec = np.concatenate((vec, rest), axis=1)
    with_features.append((comtext, big_vec))
labels = ['Comment', 'Vector']
df2 = pd.DataFrame.from_records(with_features, columns=labels)
df2.to_pickle('5.5._Word2vec_experiments/datasets/todayilearned')
print(df2.to_string(), file=open('5.5._Word2vec_experiments/datasets/todayilearned.txt', 'w'))