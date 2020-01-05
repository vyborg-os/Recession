# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:43:27 2019
@Author: vhk
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB #importing GuassianNB from naive bayes in scikit-learn
# from sklearn.svm import SVC #import Support vector classifier SVC from SVM in scikit-learn
from sklearn.model_selection import train_test_split
mydataset = pd.read_excel("C:\\Users\\Vyborg\\Desktop\\MyProject\\corrected\\cdata.xlsx")
print(mydataset.head())
print(mydataset[1:])
print(mydataset.shape)
print(mydataset.groupby("Recession").size())
summary = mydataset.describe()
print(summary)
mydataset = mydataset.values
print(mydataset)
X = mydataset[:,:-1]
Y = mydataset[:,-1]
print(X.shape) 
print(Y.shape)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state = 25)
print(x_test.shape)
print(x_train.shape)
print(y_train.shape)
print(y_test.shape)
print("")
print(x_test)
print("")
print(x_train)

#Training and test NB
clf1 = GaussianNB()
clf1.fit(x_train,y_train)
predict1 = clf1.predict(x_test)
percent1 = accuracy_score(y_test,predict1)
percent = percent1 * 100
print("")
print("Naive Bayes score: ", percent1, "or", int(percent),"%")
print("")
print("Confusion_matrix:\n", confusion_matrix(y_test,predict1))
print("")
print("Classification report:\n",classification_report(y_test,predict1))

Now training and test SVC
clf2 = SVC()
clf2.fit(x_train,y_train)
predict2 = clf2.predict(x_test)
print("SVC score: ", accuracy_score(y_test,predict2))
print("confusion_matrix: ", confusion_matrix(y_test,predict2))

print("")
plt.plot(y_test, predict1)
plt.show()
plt.plot(percent,percent1)
plt.show()
print("")
print('PREDICTING SITUATIONAL ECONOMIC RECESSION IN NIGERIA USING NAIVE BAYES')

