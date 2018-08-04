# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:51:28 2018

@author: shuli
"""
from sklearn.linear_model import LogisticRegression

def colicSklearn():
    frTrain = open('horseColicTraining.txt','r')
    frTest = open('horseColicTest.txt','r')
    trainSet = [];trainLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        trainLabel.append(float(currLine[21]))
    testSet = [];testLabel = []
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabel.append(float(currLine[21]))
    classifier = LogisticRegression(solver='liblinear',max_iter=10).fit(trainSet,trainLabel)
    test_accuracy = classifier.score(testSet,testLabel)*100
    print("正确率：%f%%"%test_accuracy)
    
if __name__ == '__main__':
    colicSklearn()