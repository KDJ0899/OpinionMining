import numpy as np

def generator(x):

    i = 0

    while i < len(x):
        yield x[i]

        i += 1

train_docs = np.load('datas/train.npy', allow_pickle=True)
# test_docs=np.load('test.npy', allow_pickle=True)

tokens = [t for d in train_docs for t in d[0]]

import nltk

text = nltk.Text(tokens, name='NMSC')
  
# selected_words = np.load('selected_words.npy', allow_pickle=True)
selected_words = [f[0] for f in text.vocab().most_common(4000)]
 
print("Start")
np.save('selected_words', selected_words)
print("close")
# def term_exists(doc):
#     return {'exists({})'.format(word): (word in set(doc)) for word in generator(selected_words)}
# # 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
#   
# # train_docs = train_docs[:50000]
# train_xy = [(term_exists(d), c) for d, c in generator(train_docs)]
# # test_xy = [(term_exists(d), c) for d, c in test_docs]
#  
# classifier = nltk.NaiveBayesClassifier.train(train_xy)
# 
# # print(nltk.classify.accuracy(classifier, test_xy))
# # => 0.80418
# # classifier.show_most_informative_features(10)
# 
# 
# 
#  
# import pickle
#  
# print("Start")
# f = open('my_classifier.pickle', 'wb')
# pickle.dump(classifier, f)
# f.close()
# print("End")