import csv
import numpy as np
import numpy.linalg as ln
import math as m
import random
from sentic import SenticPhrase
import nltk.data
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *
from nltk.corpus import wordnet as wn, stopwords
import multiprocessing as mp
from multiprocessing import Pool
from nltk.tag.stanford import StanfordPOSTagger
import pandas as pd

def preprocess(text):
    nostop_words = []
    nostop_sents = []
    withstops_words = []
    withstops_sents = []
    sentences = sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in sentences:
        sent = [word for word in tokenizer.tokenize(sentence) if word not in stopwords.words('english')]
        tokens = tokenizer.tokenize(sentence)
        nostop_words.extend(sent)
        nostop_sents.append(sent)
        withstops_words.extend(tokens)
        withstops_sents.append(tokens)
    return nostop_words, nostop_sents, withstops_words, withstops_sents

def sentyki(wordlist):
    sp = SenticPhrase(wordlist)
    try:
        text_aptitude = sp.info(wordlist)['sentics']['aptitude']
    except KeyError:
        text_aptitude = 0
    try:
        text_pleasantness = sp.info(wordlist)['sentics']['pleasantness']
    except KeyError:
        text_pleasantness = 0
    try:
        text_attention = sp.info(wordlist)['sentics']['attention']
    except KeyError:
        text_attention = 0
    try:
        text_sensitivity = sp.info(wordlist)['sentics']['sensitivity']
    except KeyError:
        text_sensitivity = 0
    return text_aptitude, text_attention, text_pleasantness, text_sensitivity #it's a list

def sentyment(wordlist):
    sp = SenticPhrase(wordlist)
    if sp.info(wordlist)['sentiment'] == 'strong negative':
        text_sentiment = -2
    elif sp.info(wordlist)['sentiment'] == 'weak negative':
        text_sentiment = -1
    elif sp.info(wordlist)['sentiment'] == 'neutral':
        text_sentiment = 0
    elif sp.info(wordlist)['sentiment'] == 'weak positive':
        text_sentiment = 1
    elif sp.info(wordlist)['sentiment'] == 'strong positive':
        text_sentiment = 2
    return text_sentiment

def relate_score(wordlist):
    first_person = ['i', 'me', 'we', 'us', 'my', 'mine', 'ours', 'our']
    relate_list = []
    for item in wordlist:
        item = item.lower()
        if item in first_person:
            relate_list.append(item)
    try:
        rel_score = len(relate_list)/len(wordlist)
    except ZeroDivisionError:
        rel_score = 0
    return rel_score

def imperatives(text):
    i = 0
    taggedlist = list(pos_tag(word_tokenize('. ' + text.lower())))
    for x in range(len(taggedlist) - 1):
        precede = taggedlist[x]
        follow = taggedlist[x + 1]
        if (precede[1] == '.' or precede[1] == 'CC' or precede[1] == 'IN') and (follow[1] == 'VB' or follow[1] == 'VBP'):
            i = i + 1
        if (precede[0] == 'please') and (follow[1] == 'VB'):
            i = i + 1
        if (precede[0] == 'you' or precede[0] == 'OP') and (follow[1] == 'MD'):
            i = i + 1
        try:
            if (precede[0] == 'why') and (follow[0] == 'don\'t') and (taggedlist[x+2][0] == 'you'):
                i = i+1
        except IndexError:
            pass
        if precede[0] == '?':
            i = i-1
    all = len(taggedlist)
    try:
        score = i/(all/2)
    except ZeroDivisionError:
        score = 0
    return score

def specificity(withstops_sents):
    text_SASD = 0
    text_SASH = 0
    text_STOC = 0
    text_SCNE = 0
    text_SCPN = 0
    text_SLEN = 0
    postags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    for sent in withstops_sents:
        ne_list = []
        pn_list = []
        nostops = [word for word in sent if word not in stopwords.words('english')]
        depth_sum = 0
        height_sum = 0
        postags_count = 0
        occur_sum = 0
        for chunk in ne_chunk(pos_tag(sent)):
            if hasattr(chunk, 'label'):
                ne_list.append(chunk)
        for tagged in pos_tag(nostops):
            if (tagged[1] == 'NNP') or (tagged[1] == 'NNPS') or (tagged[1] == 'CD'):
                pn_list.append(tagged[0])
            if tagged[1] in postags:
                postags_count = postags_count + 1
                word = tagged[0]
                syn = wn.synsets(word)
                hypo_sim = []
                hyper_sim = []
                for sense in syn:
                    hypernyms = sense.hypernyms()
                    hyponyms = sense.hyponyms()
                    for item in hypernyms:
                        simil = 1 / (sense.path_similarity(item))
                        hyper_sim.append(simil)
                    try:
                        depth = max(hyper_sim)
                        depth_sum = depth_sum + depth
                    except ValueError:
                        pass
                    for item in hyponyms:
                        simill = 1 / (sense.path_similarity(item))
                        hypo_sim.append(simill)
                    try:
                        height = max(hypo_sim)
                        height_sum = height_sum + height
                    except ValueError:
                        pass
                lemmas = wn.lemmas(word)
                occurs = []
                for lem in lemmas:
                    occurs.append(lem.count())
                occur_low = sorted(occurs, reverse=False)
                suma = 0
                for item in occur_low[:3]:
                    suma = suma+item
                occur_sum = occur_sum + suma
        try:
            SASD = depth_sum/postags_count
            SASH = height_sum/postags_count
        except ZeroDivisionError:
            SASD = 0
            SASH = 0
        STOC = occur_sum
        SCNE = len(ne_list)
        SCPN = len(pn_list)
        SLEN = len(nostops)
        text_SASD = text_SASD + SASD
        text_SASH = text_SASH + SASH
        text_STOC = text_STOC + STOC
        text_SCNE = text_SCNE + SCNE
        text_SCPN = text_SCPN + SCPN
        text_SLEN = text_SLEN + SLEN
    return text_SASD/100, text_SASH/100, text_STOC/100, text_SCNE/10, text_SCPN/10, text_SLEN/100

def calculate_features(text):
    nostop_words, nostop_sents, withstops_words, withstops_sents = preprocess(text)
    aptitude, attention, pleasantness, sensitivity = sentyki(' '.join(nostop_words))
    sentiment = sentyment(' '.join(nostop_words))
    relate = relate_score(withstops_words)
    imperative = imperatives(text)
    TASD, TASH, TTOC, TCNE, TCPN, TLEN = specificity(withstops_sents)
    return aptitude, attention, pleasantness, sensitivity, sentiment, relate, imperative, TASD, TASH, TTOC, TCNE, TCPN, TLEN

def create_matrix(name):
    z = 0
    list_of_vectors = []
    filename = ''.join('5.3._Initial_experiments/datasets/'+name+'.csv')
    print(filename)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            z = z+1
            print(z)
            text = row['Com_text']
            aptitude, attention, pleasantness, sensitivity, sentiment, relate, imperative, TASD, TASH, TTOC, TCNE, TCPN, TLEN = calculate_features(text)
            x1 = aptitude
            x2 = attention
            x3 = pleasantness
            x4 = sensitivity
            x5 = sentiment
            x6 = relate
            x7 = imperative
            x8 = TASD
            x9 = TASH
            x10 = TTOC
            x11 = TCNE
            x12 = TCPN
            x13 = TLEN
            vector = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]
            list_of_vectors.append(vector)
    X = np.matrix(list_of_vectors)
    xfile = name+'_X'
    np.save(xfile, X)

for name in ['getdisciplined_training_set', 'getdisciplined_test_set', 'reladv_training_set', 'reladv_test_set', 'pics_train', 'pics_test', 'til_train', 'til_test']:
    create_matrix(name)
