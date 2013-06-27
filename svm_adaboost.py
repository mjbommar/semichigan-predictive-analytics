# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:03:35 2013

@author: numbersmom
"""

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import math
from sklearn import preprocessing
    
def read_file(name):
    import csv
    csv_file_object=csv.reader(open(name))
    data=[]
    for row in csv_file_object:data.append(row)
    data=np.array(data)
    return data
    
train=read_file('C:/Users/numbersmom/Documents/R/sonar_train.csv')
test=read_file('C:/Users/numbersmom/Documents/R/sonar_test.csv')

y_train=train[:,60]
x_tr=train[:,0:59]
y_test=test[:,60]
x_te=test[:,0:59]
x_tr,x_te=x_tr.astype(float),x_te.astype(float)
y_train,y_test=y_train.astype(int),y_test.astype(int)
x_train,x_test= preprocessing.scale(x_tr),preprocessing.scale(x_te)
train_error=np.empty(2)
test_error=np.empty(2)
F=np.zeros(len(y_train))
F_test=np.zeros(len(y_test))
w=np.empty(len(y_train))


# This code is simply to run a svm model on the sonar data.
# Results are as follows:
#   score on the preprocessed training data is 1
#   score on the unpreprocessed training data is .538
#   score on the preprocessed test data is .856
#   score on the unpreprocessed test data is .59


clf=svm.SVC(kernel='rbf', C=10, gamma=.01)
clf.fit(x_train,y_train)
score=clf.score(x_train,y_train)
score1=clf.score(x_tr,y_train)
score2=clf.score(x_te,y_test)
score3=clf.score(x_test,y_test)
print "Score on svm model on preprocessed data is", score
print "Score on the svm model on as is data is", score1
print "Test score on pre", score3, "on as is", score2


#This starts the iterations for adaboost. 
# In the first iteration when the weights are all equal,
# the score on the preprocessed data with weights is .654


for i in range(2):
    w=np.exp(-y_train*F)
    w=w/sum(w)

    clf.fit(x_train,y_train, sample_weight=w)
    print "Tr Score on svm model with pre data and weights", clf.score(x_train, y_train)
    print "Te Score on svm model with weights", clf.score(x_test,y_test)
    g=clf.predict(x_train)
    g_test= clf.predict(x_test)
 
    print w*np.abs(.5*(g-y_train))
    e=sum(w*np.abs(.5*(g-y_train)))
    print "The error on the ", i, "th iteration is", e
    alpha=.5*np.log((1-e)/e)
    print "The alpha on the ", i, "the interation is", alpha
    F=F+alpha*g
    F_test=F_test+alpha*g_test
    train_error[i]=1-clf.score(x_train,y_train)
    test_error[i]=1-clf.score(x_test,y_test)
    print train_error[i],test_error[i]
    
    

