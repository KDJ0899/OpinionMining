'''
Created on 2018. 11. 18.

@author: user
'''

def read_data(filename):
    with open(filename, 'rt', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외
    return data
# train_data = read_data('ratings_train.txt')
test_data = read_data('datas/ratings_test.txt')

from konlpy.tag import Okt
pos_tagger = Okt()
def tokenize(doc):
    #norm = 문장을 정규화, stem = 각 단어에서 어간 추출, join = 형태소와 품사를 같이 붙일지 정함.
    return [t for t in pos_tagger.pos(doc, norm=True, stem=True, join=False)]
# train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
# 잘 들어갔는지 확인
print("Start")
import numpy as np
# np.save('train', train_docs)
np.save('datas/test2', test_docs)
print("close")

    