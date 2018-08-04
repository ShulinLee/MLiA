# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 11:05:36 2018

@author: ShulinLee
"""

from math import log
import operator
import pickle

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #分类属性
    return dataSet, labels  

def calEnt(dataSet):
    num = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        currLabel = featVec[-1]
        labelCount[currLabel] = labelCount.get(currLabel,0)+1
    ent = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/num
        ent -= prob*log(prob,2)
    return ent

def splitData(dataSet,axis,value):
    retData = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retData.append(reducedFeatVec)
    return retData

def chooseBestFeature(dataSet):
    numFeat = len(dataSet[0])-1
    baseEnt = calEnt(dataSet)
    bestInfoGain = 0.0
    bestFeat = -1
    for i in range(numFeat):
        featList = [example[i] for example in dataSet]
        uniqValue = set(featList)
        newEnt = 0.0
        for value in uniqValue:
            subDataSet = splitData(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEnt += prob*calEnt(subDataSet)
        infoGain = baseEnt-newEnt
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels,featLabels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqValues = set(featValues)
    for value in uniqValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitData(dataSet,bestFeat,
              value),subLabels,featLabels)
    
    return myTree
        
def classify(inputTree,featLabels,testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
    
def storeTree(inputTree,filename):
    with open(filename,'wb') as fw:
        pickle.dump(inputTree,fw)
        
def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)    