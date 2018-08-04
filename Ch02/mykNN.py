# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:02:19 2018

@author: shulinli
"""

import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines

def classify0(inX,dataSet,labels,k):
    #计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    squareSum = sum(diffMat**2,axis=1)
    dist = np.sqrt(squareSum)
    #找出k个最近点
    classCount = {}
    distIndex = np.argsort(dist)
    for i in range(k):
        voteLabel = labels[distIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    #找出频率最高的label
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename,'r')
    lines = fr.readlines()
    numOlines = len(lines)
    returnMat = np.zeros((numOlines,3))
    labelVector = []
    index =0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        labelVector.append(eval(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat,labelVector

def showdata(datingDataMat,datingLabels):
    font = fm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    fig = plt.figure()
    axs = fig.add_subplot(111)
    labelColor = []
    for i in datingLabels:
        if i == 1:
            labelColor.append('black')
        elif i == 2:
            labelColor.append('orange')
        elif i == 3:
            labelColor.append('red')
    axs.scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=labelColor,s=15,alpha=0.5)
    axs0_title = axs.set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel = axs.set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel = axs.set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    plt.setp(axs0_title,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel,size=9,weight='bold',color='black')
    plt.setp(axs0_ylabel,size=9,weight='bold',color='black')
    didntLike = mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntLike')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',markersize=6,label='smallDoses')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='largeDoses')
    axs.legend(handles=[didntLike,smallDoses,largeDoses])
    plt.show()
def autoNorm(dataSet):    
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal-minVal
    normDataSet = (dataSet-minVal)/ranges
    return normDataSet,minVal,ranges

def datingClassTest():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        print("The classifier came back with: %d,the real answer is :%d"\
              %(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("The total error rate is:%f"%(errorCount/numTestVecs))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = eval(input("percentage of time spent playing video games?"))
    ffMiles = eval(input("frequent flier miles earned per year?"))
    iceCream = eval(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([percentTats,ffMiles,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:  ",resultList[classifierResult-1])
    
if __name__ == '__main__':
    classifyPerson()