# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:37:06 2018

@author: shit
"""
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def classify(inX,weight):
    prob = sigmoid(sum(inX*weight))
    if prob >0.5:
        return 1.0
    else:
        return 0.0

def stocGradAscent(data,label,iterNum = 150):
    dataM = np.array(data)
    m,n = np.shape(data)
    weight = np.ones(n)
    for i in range(iterNum):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4.0/(1.0+i+j)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataM[randIndex]*weight))
            error = label[randIndex] - h
            weight = weight + alpha*error*dataM[randIndex]
            del(dataIndex[randIndex])
    return weight
    
def testModel():
    frTrain = open('horseColicTraining.txt','r')
    frTest = open('horseColicTest.txt','r')
    trainSet = [];labelSet = []
    for line in frTrain:
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        labelSet.append(float(currLine[21]))
    trainWeight = stocGradAscent(trainSet,labelSet)
    errorCount = 0.0;testNum = 0.0
    for line in frTest:
        testNum += 1
        testArr = []
        currLine = line.strip().split('\t')
        for i in range(21):
            testArr.append(float(currLine[i]))
        if int(classify(testArr,trainWeight)) != int(currLine[21]):
            errorCount += 1
    errorRate = errorCount/testNum
    print("The error rate of this test is :%.2f"%(errorRate))
    return errorRate

def multiTest():
    numTest =10;errorSum = 0.0
    for k in range(numTest):
        errorSum += testModel()
    print("After %d iterations the average error rate is: %.2f"%(numTest,errorSum/float(numTest)))