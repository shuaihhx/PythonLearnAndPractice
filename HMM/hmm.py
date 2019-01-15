# -*- coding: utf-8 -*-

'''
@Description: HMM
@Author: zhanglieshuai
@Date: 2019-01-13 15:04:34
@LastEditTime: 2019-01-15 09:42:48
'''
from collections import defaultdict
import pickle
import numpy as np

class MyHMM(object):

    def __init__(self, data):
        self.wordNum = 1000
        self.tagList = ('B', 'M', 'E', 'S')
        self.tag2id = {tag: i for i, tag in enumerate(self.tagList)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}
        self.initMat = np.zeros([len(self.tagList)])
        self.transMat = np.zeros([len(self.tagList), len(self.tagList)])
        self.emitMat = np.zeros([len(self.tagList),self.wordNum])
        self.word2id = dict()
        self.id2word = dict()
        self.static(data)
        self.buildHMM()

    def static(self, data):
        
        for sentence in data:
            if len(sentence[0]) < 2:
                continue

            i = -1
            preTag = None
            for word, tag in zip(sentence[0], sentence[1]):
                i += 1
                if not word in self.word2id:
                    self.word2id[word] = len(self.word2id)
                if len(self.word2id) > self.wordNum:
                    self.emitMat = np.hstack((self.emitMat, np.zeros([len(self.tagList),1000])))
                    self.wordNum += 1000

                self.emitMat[self.tag2id[tag], self.word2id[word]] += 1

                if i == 0:
                    self.initMat[self.tag2id[tag]] += 1
                    
                else:
                    self.transMat[self.tag2id[preTag], self.tag2id[tag]] += 1
                preTag = tag
        self.wordNum = len(self.word2id)
        self.emitMat = self.emitMat[:,0:self.wordNum]
        self.id2word = {i: word for word, i in self.word2id.items()}

    def buildHMM(self):
        self.initProb = self.initMat / self.initMat.sum()

        self.transMat = self.transMat.T
        self.transProb = self.transMat / self.transMat.sum(axis=0)
        self.transProb = self.transProb.T

        self.emitMat = self.emitMat.T
        self.emitProb = self.emitMat / self.emitMat.sum(axis=0)
        self.emitProb = self.emitProb.T

    def viterbi(self, text):
        n = len(text)
        probMat = np.zeros([n, len(self.tagList)])
        memoryMat = np.zeros([n, len(self.tagList)], dtype=np.int16)

        
        for i, char in enumerate(text):
            if i == 0:
                for j in self.id2tag:
                    probMat[i, j] = self.initProb[j]
                    if char in self.word2id:
                        probMat[i, j] *= self.emitProb[j, self.word2id[char]]
            else:
                for j in self.id2tag:
                    tempProb = 0
                    tempIndex = -1
                    for m in self.id2tag:
                        prob = self.transProb[m, j] * probMat[i-1, m] 
                        if char in self.word2id:
                            prob *= self.emitProb[j, self.word2id[char]]
                        if prob > tempProb:
                            tempProb = prob
                            tempIndex = m

                    probMat[i, j] = tempProb
                    memoryMat[i, j] = tempIndex
        k = 0
        maxNum = 0
        for index in self.id2tag:
            if probMat[n-1, index] > maxNum:
                maxNum = probMat[n-1, index]
                k = index
        outTag = list()
        outTag.append(self.id2tag[k])
        for i in range(1, n):
            outTag.append(self.id2tag[memoryMat[n-i, k]])
            k = memoryMat[n-i ,k]
        return outTag



    def decode(self, text):
        outTag = self.viterbi(text)
        outTag.reverse()
        outText = ''
        for i, tag in enumerate(outTag):
            if tag == 'S' or tag == 'E':
                outText += text[i] + ' '
            else:
                outText += text[i]

        if outText[-1] == 'S' or outText[-1] == 'E':
            return outText[:-1]
        else:
             return outText



def trainHmm():
    with open('data','rb') as f:
        data = pickle.load(f)

    hmm = MyHMM(data)

    with open('hmm', 'wb') as f:
        pickle.dump(hmm, f)

def hmmPredict():
    with open('hmm', 'rb') as f:
        hmm = pickle.load(f)

    text = '小明硕士毕业于中国科学院计算所'
    result = hmm.decode(text)
    print(result)
    # result : 小明 硕士 毕业 于 中国 科学院 计算 所

if __name__ == '__main__':
    hmmPredict()


    
