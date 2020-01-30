'''
Created on 2020. 1. 30.

@author: kxv13
'''
import pickle
import numpy as np
import nltk

def generator(x):

    i = 0

    while i < len(x):
        yield x[i]

        i += 1
test_docs=np.load('datas/test.npy', allow_pickle=True)

tokens = [t for d in test_docs for t in d[0]]

classifier_f = open("datas/my_classifier.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

selected_words = np.load('datas/selected_words.npy', allow_pickle=True)



def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in generator(selected_words)}
# 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
   
test_xy = [(term_exists(d), c) for d, c in test_docs]
  
print(nltk.classify.accuracy(classifier, test_xy))
# => 0.80418
classifier.show_most_informative_features(10)