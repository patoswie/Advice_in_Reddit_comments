import urllib
from urllib import request
import json
import time
import nltk, re, pprint
import codecs
import numpy as np
from collections import Counter
import random
from sentic import SenticPhrase
import nltk.data
from nltk import sent_tokenize, ne_chunk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import wordnet as wn, stopwords
import multiprocessing as mp
from multiprocessing import Pool
from nltk.tag import pos_tag
from nltk.tag.stanford import StanfordPOSTagger
import pandas as pd
import re

def preprocess(text):
    text = re.sub(r"\(*http\S+", "[link]", text) #removes links
    nostop_words = []
    nostop_sents = []
    withstops_words = []
    withstops_sents = []
    sentences = sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in sentences:
        sent = [word.lower() for word in tokenizer.tokenize(sentence) if word.lower() not in stopwords.words('english')]
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
    return text_aptitude, text_attention, text_pleasantness, text_sensitivity

def relate_score(withstops_words):
    first_person = ['i', 'me', 'we', 'us', 'my', 'mine', 'ours', 'our']
    relate_list = []
    for item in withstops_words:
        item = item.lower()
        if item in first_person:
            relate_list.append(item)
    try:
        rel_score = len(relate_list)/len(withstops_words)
    except ZeroDivisionError:
        rel_score = 0
    return rel_score

def imperatives(text):
    i = 0
    taggedlist = list(pos_tag(nltk.word_tokenize('. ' + text.lower())))
    punct = ['.', ',', '``', ':']
    for x in range(len(taggedlist) - 1):
        precede = taggedlist[x]
        follow = taggedlist[x + 1]
        if (precede[1] in punct or precede[1] == 'CC' or precede[1] == 'IN') and (follow[1] == 'VB' or follow[1] == 'VBP') and (follow[0] != 'do' or follow[0] != 'have'):
            i = i + 1
        if (precede[0] == 'please') and (follow[1] == 'VB'):
            i = i + 1
        if (precede[0] == 'you' or precede[0] == 'OP') and (follow[1] == 'MD'):
            i = i + 1
    all = len(taggedlist)
    try:
        score = i/(all/2)
    except ZeroDivisionError:
        score = 0
    return score

def advice_express(text):
    score = 0
    phrases = ['you need to', 'op needs to', 'you have to', 'op has to', 'it might be worth', 'i would', 'it would be good to', 'it might be good to', 'you had better', 'you\'d better', 'your only option is', 'why don\'t you', 'have you thought about', 'have you tried', 'how about', 'if i were you', 'recommend', 'suggest', 'advise', 'you could always', 'have you considered', 'why not', '[link]']
    for phrase in phrases:
        if phrase in text.lower():
            score = score+1
    return score/10

def specificity(withstops_sents):
    text_SASD = 0
    text_SASH = 0
    text_SCNE = 0
    text_SCPN = 0
    text_STOC = 0
    text_LEN = 0
    postags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    for sent in withstops_sents:
        ne_list = []
        pn_list = []
        nostops = [word.lower() for word in sent if word.lower() not in stopwords.words('english')]
        depth_sum = 0
        height_sum = 0
        postags_count = 0
        occur_all = []
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
                        height = min(hypo_sim)
                        height_sum = height_sum + height
                    except ValueError:
                        pass
                lemmas = wn.lemmas(word)
                occurs = []
                for lem in lemmas:
                    occurs.append(lem.count())
                occur_low = sorted(occurs, reverse=False)
                try:
                    occur_all.append(occur_low[0])
                except IndexError:
                    pass
        try:
            SASD = depth_sum/postags_count
            SASH = height_sum/postags_count
        except ZeroDivisionError:
            SASD = 0
            SASH = 0
        STOC = sum(sorted(occur_all, reverse=False)[:3])
        SCNE = len(ne_list)
        SCPN = len(pn_list)
        LEN = len(nostops)
        text_SASD = text_SASD + SASD
        text_SASH = text_SASH + SASH
        text_STOC = text_STOC + STOC
        text_SCNE = text_SCNE + SCNE
        text_SCPN = text_SCPN + SCPN
        text_LEN = text_LEN + LEN
    ASHD = text_SASD - text_SASH
    return text_SASD/100, text_SASH/100, ASHD/100, text_STOC/100, text_SCNE/10, text_SCPN/10, text_LEN/100

def calculate_features(text):
    nostop_words, nostop_sents, withstops_words, withstops_sents = preprocess(text)
    aptitude, attention, pleasantness, sensitivity = sentyki(' '.join(nostop_words))
    relate = relate_score(withstops_words)
    imperative = imperatives(text)
    advice = advice_express(text)
    ASD, ASH, ASHD, TTOC, CNE, CPN, LEN = specificity(withstops_sents)
    return aptitude, attention, pleasantness, sensitivity, relate, imperative, advice, ASD, ASH, ASHD, TTOC, CNE, CPN, LEN

def get_comments(redditname):
    print(redditname)
    dataframe = pd.read_pickle(redditname)
    comment_list = dataframe['Comment'].tolist()
    data = []
    for i in range(len(comment_list)):
        com_text = comment_list[i]
        aptitude, attention, pleasantness, sensitivity, relate, imperative, advice, ASD, ASH, ASHD, TTOC, CNE, CPN, LEN = calculate_features(com_text)
        line = [com_text, aptitude, attention, pleasantness, sensitivity, relate, imperative, advice, ASD, ASH, ASHD, TTOC, CNE, CPN, LEN]
        data.append(line)

    labels = ['Comment', 'Aptitude', 'Attention', 'Pleasantness', 'Sensitivity', 'Relatability', 'Imperative', 'Advice', 'ASD', 'ASH', 'ASHD', 'TOC', 'CNE', 'CPN', 'LEN']
    df = pd.DataFrame.from_records(data, columns=labels)
    df.to_pickle(redditname)
    print(df.to_string(), file=open(redditname + '.txt', 'w'))

redditnames = [''.join('Advice'), ''.join('getdisciplined')]
pool = mp.Pool(processes=mp.cpu_count())
processes = [pool.apply_async(get_comments, args=(redditname,)) for redditname in redditnames]
for p in processes:
    result = p.get()

#now move on to calculating_word2vec_features.py code