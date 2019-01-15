# -*- coding: utf-8 -*-

'''
@Description: process the corpus, to BEMS(for word segmentation)
@Author: zhanglieshuai
@Date: 2019-01-13 15:04:49
@LastEditTime: 2019-01-14 11:14:48
'''
import os
import pickle

path = r'D:\人民日报_细粒度'

def getData():
    data = list()
    for root, _, files in os.walk(path):
        for file in files:
            data.extend(fileProcess(os.path.join(root,file)))
    return data

def fileProcess(file):
    fileData = list()
    with open(file, 'r' ,encoding='utf8') as f:
        
        for line in f:
            sentence = list()
            for word in line.strip().split():
                index = word.rfind('/')
                if index < 1:
                    break
                if word[:index] == '钟汉良':
                    print(line)
                sentence.append(word[:index])
            a = sentenceProcess(sentence)
            fileData.append(a)
    return fileData

def sentenceProcess(sentence):
    data = list()
    tag = list()
    for word in sentence:
        if len(word) == 1:
            data.append(word)
            tag.append('S')
        else:
            data.extend([char for char in word])
            tag.append('B')
            tag.extend('M'*(len(word)-2))
            tag.append('E')
    return data,tag

if __name__ == '__main__':
    data = getData()








