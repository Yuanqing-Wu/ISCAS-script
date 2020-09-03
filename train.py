import pandas as pd
import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import feature_analysis as fa

def feature_train(train, test, feature, train_size = 0, save_mode = None, C = 100000, gamma = 0):
    
    X_train = train.loc[:, feature]
    X_test = test.loc[:, feature]

    y_train = train.loc[:, 'mode']
    y_test = test.loc[:, 'mode']

    y_train = y_train.copy()
    y_test = y_test.copy()

    y_train[y_train != 2000] = 1
    y_train[y_train == 2000] = 0

    y_test[y_test != 2000] = 1
    y_test[y_test == 2000] = 0

    if train_size != 0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size = train_size, random_state = 1, stratify = y_train)

    start_time = time.time()

    if gamma == 0:
        svc = svm.SVC(kernel='rbf', C = C, probability = True)
    else:
        svc = svm.SVC(kernel='rbf', C = C, gamma = gamma, probability = True)

    svc.fit(X_train, y_train)

    end_time = time.time()
    if save_mode != None:
        joblib.dump(svc, save_mode + '.model')  
        

    print(feature, svc.score(X_test, y_test))   
    print(svc.support_.shape)
    
    print("time:%d"  % (end_time-start_time)) 
    print()


def paraments_search(df_train, df_test, feature, C, gamma):

    X_train = df_train.loc[:, feature]
    X_test = df_test.loc[:, feature]

    y_train = df_train.loc[:, 'mode']
    y_test = df_test.loc[:, 'mode']

    y_train = y_train.copy()
    y_test = y_test.copy()

    y_train[y_train != 2000] = 1
    y_train[y_train == 2000] = 0

    y_test[y_test != 2000] = 1
    y_test[y_test == 2000] = 0

    for c in C:
        for g in gamma:
            svc = svm.SVC(kernel='rbf', C = c, gamma = g, probability = True)

            start_time = time.time()
            svc.fit(X_train, y_train)
            end_time = time.time()

            score = svc.score(X_test, y_test)
            print(c, g, score, svc.support_.shape[0], end_time-start_time)

#read data
train_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\train\\'
test_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\test\\'    

df_train, seq_name= fa.read_csv_data(train_set_path)
df_test, seq_name= fa.read_csv_data(test_set_path)

print("train set shape: ", df_train.shape[0])
print("test set shape: ", df_test.shape[0])

# ['w', 'qp', 'nvar', 'H', 'ngradx', 'ngrady', 'gmx', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']


# feature_train(df_train, df_test, ['w', 'qp', 'nvar', 'H', 'ngradx', 'ngrady', 'gmx', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
feature_train(df_train, df_test, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'], gamma = 2e-07, save_mode='s_ns_32x32')
# paraments_search(df_train, df_test, ['qp', 'nvar', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'], [100000], [0.0000000005, 0.0000000001, 0.00000000005])
