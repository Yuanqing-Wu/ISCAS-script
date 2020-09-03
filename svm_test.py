import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import feature_analysis as fa

def rdo_th(clf, X_test, th):
    
    num_error = 0
    y_pre = clf.predict(X_test)
    y_pre_prob = clf.predict_proba(X_test)
    th = 0
    num_pre = 0
    for i in range(y_pre.shape[0]):
        if y_pre_prob[i][0] > th or y_pre_prob[i][0] < (1 - th):
            if y_pre[i] != y_test[i]:
                num_error = num_error + 1
            num_pre = num_pre + 1
        accurency = 1 - num_error/y_pre.shape[0]
    print(y_pre.shape[0], num_pre, accurency)

test_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\modeltest\\' 
df_test, seq_name= fa.read_csv_data(test_set_path)  

#df_test = df_test[df_test['w'] == df_test['h']]
#df_test = df_test[df_test['w'] != 8]

X_test = df_test.loc[:, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]
y_test = df_test.loc[:, 'mode']
y_test[y_test != 2000] = 1
y_test[y_test == 2000] = 0

clf = joblib.load('square_64_32_16.model')
#print(clf.score(X_test, y_test))
y_pre = clf.predict(X_test)
y_pre_prob = clf.predict_proba(X_test)

prelog = open('pre.csv', 'w')
for i in range(df_test.shape[0]):
    prelog.write(str(df_test.loc[i, 'poc']) + ',' + str(df_test.loc[i, 'x']) + ',' + str(df_test.loc[i, 'y']) + ',' + str(df_test.loc[i, 'w']) + ',' + str(df_test.loc[i, 'h']) + ',' + str(y_test[i]) + ',' + str(y_pre[i]) + ',' + str(y_pre_prob[i][0]) + ',' + str(y_pre_prob[i][1]) + '\n')

prelog.close()
