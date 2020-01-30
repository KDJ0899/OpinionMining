'''
Created on 2020. 1. 28.

@author: kxv13
'''
import numpy as np


def generator1(x):

    i = 0

    while i < len(x):
        yield x[i][0],x[i][1]

        i += 1
def generator(x):

    i = 0

    while i < len(x):
        yield x[i]

        i += 1

# train_docs = np.load('train.npy', allow_pickle=True)
test_docs=np.load('datas/test2.npy', allow_pickle=True)

tokens = [t for d in test_docs for t in d[0]]

import nltk

text = nltk.Text(tokens, name='NMSC')
import konlpy

# selected_words = np.load('selected_words.npy', allow_pickle=True)
selected_words = [f[0]  for f in text.vocab().most_common(100)]

print(selected_words)
 