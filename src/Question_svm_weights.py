# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:47:30 2013

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
scaler=preprocessing.StandardScaler().fit(x_tr)
x_tr_scaled=scaler.transform(x_tr)
x_te_scaled=scaler.transform(x_te)

w=np.ones(len(y_train))

clf=svm.SVC(kernel='rbf', C=10, gamma=.01)
clf.fit(x_tr,y_train)
score_un_tr=clf.score(x_tr,y_train)
score_un_test=clf.score(x_te,y_test)
clf.fit(x_tr_scaled,y_train)
score_scaled_tr=clf.score(x_tr_scaled,y_train)
score_scaled_test=clf.score(x_te_scaled,y_test)

print "SVM training score unscaled", score_un_tr, "test score unscaled", score_un_test
print "SVM training score scaled", score_scaled_tr, "test scored scaled", score_scaled_test

w=w/sum(w)
clf1=svm.SVC(kernel='rbf', C=10/float(len(y_train)), gamma=.01, probability=True )
clf1.fit(x_tr_scaled,y_train,sample_weight=w)
print "Traing score with sample weights is ", clf1.score(x_tr,y_train)
print "Score with sample weights is", clf1.score(x_te_scaled,y_test)
Pr=clf1.predict_proba(x_tr_scaled)