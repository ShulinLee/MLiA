# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:47:07 2018

@author: shuli
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pdb

def file2matrix(filename):
    fr = open(filename,'r')
    lines = fr.readlines()
    numOlines = len(lines)
    returnMat = np.zeros((numOlines,3))
    labelVec = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        labelVec.append(listFromLine[-1])
        index += 1
    fr.close()
    return returnMat,labelVec

def classifier(vec,dataset,labels):
    X_train,X_test,y_train,y_test = train_test_split(dataset,labels,random_state=0)
    scale = StandardScaler().fit(X_train)
    standardized_X = scale.transform(X_train)
    standardized_X_test = scale.transform(X_test)
    standardized_vec = scale.transform(vec.reshape(1,-1))
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(standardized_X,y_train)
    #model evaluation
    y_pred = knn.predict(standardized_X_test)
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    return knn.predict(standardized_vec)

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses'] 
    ffMiles = eval(input("frequent flier miles earned per year?"))
    percentTats = eval(input("percentage of time spent playing video games?"))
    iceCream = eval(input("liters of ice cream consumed per year?"))
    vec = np.array([ffMiles,percentTats,iceCream])
    dataset,labels = file2matrix('datingTestSet2.txt')
#    pdb.set_trace()
    result = int(classifier(vec,dataset,labels))
    print('You will probably like this person: ',resultList[result-1])
    
if __name__=='__main__':
    classifyPerson()