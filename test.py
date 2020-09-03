import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import feature_analysis as fa

def rdo_th(clf, X_test, y_test, th):
    
    num_error = 0
    y_pre = clf.predict(X_test)
    y_pre_prob = clf.predict_proba(X_test)
    num_pre = 0
    for i in range(y_pre.shape[0]):
        if y_pre_prob[i][0] > th or y_pre_prob[i][0] < (1 - th):
            if y_pre[i] != y_test[i]:
                num_error = num_error + 1
            num_pre = num_pre + 1
        accurency = 1 - num_error/y_pre.shape[0]
    print(y_pre.shape[0], num_pre, accurency)
    return accurency

def plot_pre_accurency_with_th(model, df_test, feature):

    clf = joblib.load(model)

    X_test = df_test.loc[:, feature]
    y_test = df_test.loc[:, 'mode']

    y_test[y_test != 2000] = 1
    y_test[y_test == 2000] = 0

    accurency = []
    for th in np.linspace(0.5, 1, 11):
        accurency.append(rdo_th(clf, X_test, y_test, th))
    
    plt.figure(num='accurency')
    plt.plot(np.linspace(0, 1, 11), accurency, 'o-')
    plt.savefig(fname = 'accurency.png')
        

test_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\modeltest\\' 
df_test, seq_name= fa.read_csv_data(test_set_path)  

model = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\model\\s_ns_32x32.model'

plot_pre_accurency_with_th(model, df_test, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])

