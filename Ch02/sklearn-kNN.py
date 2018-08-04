# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:06:27 2018

@author: shulinli
"""
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier as kNN

def img2vec(filename):
    returnVec = np.zeros((1,1024))
    fr = open(filename,'r')
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i+j] = int(lineStr[j])
    return returnVec

def handwritingClassTest():
    #构建训练集
    hwLabels = []
    trainingSetNames = os.listdir('trainingDigits')
    m = len(trainingSetNames)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingSetNames[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vec('trainingDigits/%s'%(fileNameStr))
    #构建kNN模型并训练    
    neigh = kNN(n_neighbors=3,algorithm='auto')
    neigh.fit(trainingMat,hwLabels)
    #将模型用于测试集
    testSetNames = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testSetNames)
    for i in range(mTest):
        fileNameStr =testSetNames[i]
        classNumber = int(fileNameStr.split('_')[0])
        vecUnderTest = img2vec('testDigits/%s'%(fileNameStr))
        classifierResult = neigh.predict(vecUnderTest)
        print("分类结果是%d,真是结果是%d"%(classifierResult,classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0
    print("总共出错%d个数据，错误率为%.4f%%"%(errorCount,errorCount/mTest*100))
    
if __name__ == '__main__':
    handwritingClassTest()
    
    
    