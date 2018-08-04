# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:15:24 2018

@author: shuli
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt','r')
    for line in fr:
        linArr = line.strip().split()
        dataMat.append([1.0,eval(linArr[0]),eval(linArr[1])])
        labelMat.append(eval(linArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    weight = np.ones((n,1))
    alpha = 0.001
    maxCycles = 500
    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weight)
        error = labelMat-h
        weight = weight + alpha*dataMatrix.transpose()*error
    return weight

def plotBestFit(weight):
    dataMat,classLabel = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(classLabel[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
#    pdb.set_trace()
    x = np.arange(-3.0,3.0,0.1)
    y = np.array((-weight[0]-weight[1]*x)/weight[2]).transpose()
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
def stocGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    data = np.array(dataMatrix)
    weight = np.ones(n)
    alpha = 0.01
    for i in range(m):
        h = sigmoid(sum(data[i]*weight))
#        pdb.set_trace()
        error = classLabels[i] - h
        weight = weight + (alpha*error)*data[i]
    return weight
       
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    data = np.array(dataMatrix)
    weight = np.ones(n)
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(data[randIndex]*weight))
            error = classLabels[randIndex] - h
            weight = weight + alpha*error*data[randIndex]
            del(dataIndex[randIndex])
    return weight
            