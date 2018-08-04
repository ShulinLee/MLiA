# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 10:28:38 2018

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

def plotDataset():
    x,y = loadDataset('ex0.txt')
    n = len(x)
    xcord = [];ycord = []
    for i in range(n):
        xcord.append(x[i][1])
        ycord.append(y[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord,ycord,s=20,c='blue',alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
    
def standRegres(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

def plotRegression():
    x,y = loadDataset('ex0.txt')
    ws = standRegres(x,y)
    xMat = np.mat(x)
    yMat = np.mat(y)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')
    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,c='blue',
               alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

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
    
def plotlwlr():
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc",size=14)
    x,y = loadDataset('ex0.txt')
    yHat1 = lwlrTest(x,x,y,1.0)
    yHat2 = lwlrTest(x,x,y,0.01)
    yHat3 = lwlrTest(x,x,y,0.003)
    xMat = np.mat(x);yMat = np.mat(y)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig,ax = plt.subplots(3,1,figsize=(10,8))
    ax[0].plot(xSort[:,1],yHat1[srtInd],c='red')
    ax[1].plot(xSort[:,1],yHat2[srtInd],c='red')
    ax[2].plot(xSort[:,1],yHat3[srtInd],c='red')
    ax[0].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,
       c='blue',alpha=0.5)
    ax[1].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,
       c='blue',alpha=0.5)
    ax[2].scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20,
       c='blue',alpha=0.5)
    ax0_title_text = ax[0].set_title(u'局部加权回归曲线,k=1.0',
                       FontProperties=font)
    ax1_title_text = ax[1].set_title(u'局部加权回归曲线,k=0.01',
                       FontProperties=font)
    ax2_title_text = ax[2].set_title(u'局部加权回归曲线,k=0.003',
                       FontProperties=font)
    plt.setp(ax0_title_text,size=8,weight='bold',color='red')
    plt.setp(ax1_title_text,size=8,weight='bold',color='red')
    plt.setp(ax2_title_text,size=8,weight='bold',color='red')
    plt.xlabel('X')
    plt.show()
    
if __name__ == '__main__':
    plotlwlr()