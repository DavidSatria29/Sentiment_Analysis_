import numpy as np
import pandas as pd
import re
import string
import collections
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# case folding
def case_folding(data):
    data = data.lower()
    return data

# proses cleansing remove regex (remove special characters and numbers)
def cleansing(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

# process remove stop words using nltk
stop_words = set(stopwords.words('english'))
def set_stop_words(text):
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word not in stop_words]
    text = ' '.join(text)
    return text

def get_sentiment_dict():
    sentiment_dict = {}
    with open('web/model/data/normalized_sentiment_score.txt', 'r', encoding= 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            sentiment_dict[line[0]] = float(line[1])
    return sentiment_dict

# function Word List
def get_word_list():
    word_list = []
    with open('web/model/data/normalized_sentiment_score.txt', 'r', encoding= 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            word_list.append(line[0])
    return word_list

def get_word_vectors():
    word_list = get_word_list()
    vecs = np.loadtxt('web/model/data/word_embedding.txt')
    word_vec = {}
    for i in range(len(word_list)):
        word_vec[word_list[i]] = vecs[i]
    return word_vec

def get_weighted_word_vectors():
    word_vec = get_word_vectors()
    sentiment_dict = get_sentiment_dict()
    for i in word_vec.keys():
        if i in sentiment_dict.keys():
            word_vec[i] = sentiment_dict[i] * word_vec[i]
    return word_vec

# Get Sentence List
def list_sentences():
    sentence_list = []
    data = pd.read_csv('web/model/data/flipkart_clean.csv')
    for sentence in data['Summary']:
        sentence_list.append(sentence)
    return sentence_list

def process_data(sentence_length, words_size, embed_size):
    sentences = list_sentences()
    sentences = [str(sentence) for sentence in sentences]
    sentences = [sentence.split() for sentence in sentences]
    frequency = collections.Counter()
    for sentence in sentences:
        for word in sentence:
            if word is not None and word != 'nan':
                frequency[word] += 1

    word2index = dict()
    for i, x in enumerate(frequency.most_common(words_size)):
        word2index[x[0]] = i + 1

    word2vec = get_weighted_word_vectors()
    word_vectors = torch.zeros(words_size + 1, embed_size)
    for k, v in word2index.items():
        if k in word2vec:
            word_vectors[v, :] = torch.from_numpy(word2vec[k])
        else:
            print(f"Warning: '{k}' not in word2vec")
    rs_sentences = []
    for sentence in sentences:
        sen = []
        for word in sentence:
            if word in word2index.keys():
                sen.append(word2index[word])
            else:
                sen.append(0)
        if len(sen) < sentence_length:
            sen.extend([0 for _ in range(sentence_length - len(sen))])
        else:
            sen = sen[:sentence_length]
        rs_sentences.append(sen)
    label = [0 for _ in range(40620)]
    label.extend([1 for _ in range(6884)])
    label.extend([2 for _ in range(2496)])
    label = np.array(label)
    return rs_sentences, label, word_vectors, word2index

# get word2index
def get_word2index():
    word2index = process_data(12, 35000, 768)
    return word2index[3]