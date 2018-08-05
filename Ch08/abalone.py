# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:36:10 2018

@author: shuli
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def loadDataset(filename):
    numFeat = len(open(filename).readline().split('\t'))-1
    datMat = [];labelMat = []
    fr = open(filename,'r')
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        datMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    fr.close()
    return datMat,labelMat

def standRegres(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

def lwlr(testPoint,x,y,k=1.0):
    xMat = np.mat(x);yMat = np.mat(y).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for i in range(m):
        diffMat = testPoint-xMat[i,:]
        weights[i,i] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,x,y,k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],x,y,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

if __name__ == '__main__':
    abX,abY = loadDataset('abalone.txt')
    print("训练集与测试集相同：局部加权线性回归，核k的大小对预测的影响：")
    yHat01s = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1s = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10s = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    print("k=0.1时，误差大小为：",rssError(abY[0:99],yHat01s.T))
    print("k=1.0时，误差大小为：",rssError(abY[0:99],yHat1s.T))
    print("k=10时，误差大小为：",rssError(abY[0:99],yHat10s.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归,核k不是越小越好:')
    yHat01d = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    yHat1d = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    yHat10d = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
    print("k=0.1时，误差大小为：",rssError(abY[100:199],yHat01d.T))
    print("k=1.0时，误差大小为：",rssError(abY[100:199],yHat1d.T))
    print("k=10时，误差大小为：",rssError(abY[100:199],yHat10d.T))
    print("")
    print("训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:")
    print("k=1时，误差大小为：",rssError(abY[100:199],yHat1d.T))
    ws = standRegres(abX[0:99],abY[0:99])
    yHat = np.mat(abX[100:199]*ws)
    print("简单线性回归的误差大小为：",rssError(abY[100:199],yHat.T.A))