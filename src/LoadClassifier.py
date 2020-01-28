'''
Created on 2020. 1. 28.

@author: kxv13
'''
import pickle
import numpy as np
from konlpy.tag import Okt
pos_tagger = Okt()

def tokenize(doc):
    # norm, stem은 optional
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def generator(x):

    i = 0

    while i < len(x):
        yield x[i]

        i += 1
        
def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in generator(selected_words)}


classifier_f = open("my_classifier.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

selected_words = np.load('selected_words.npy', allow_pickle=True)
words = tokenize("나는 그저 그랬는데 친구는 좋아 하더라")
feats = term_exists(words)

print(words)
print(classifier.classify(feats))

 
