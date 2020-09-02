import pandas as pd
import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import feature_analysis as fa

def feature_train(train, test, feature, train_size = 0, save_mode = None):

    

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
    C = 100000
    #gamma = 4.45e-08
    svc = svm.SVC(kernel='rbf', C = C, probability = True)
    svc.fit(X_train, y_train)

    end_time = time.time()
    if save_mode != None:
        joblib.dump(svc, save_mode + '.model')  
        

    print(feature, svc.score(X_test, y_test))   
    print(svc.support_.shape)
    
    print("time:%d"  % (end_time-start_time)) 
    print()

    num_error = 0
    y_pre = svc.predict(X_test)
    y_pre_prob = svc.predict_proba(X_test)
    th = 0
    num_pre = 0
    for i in range(y_pre.shape[0]):
        if y_pre_prob[i][0] > th or y_pre_prob[i][0] < (1 - th):
            if y_pre[i] != y_test[i]:
                num_error = num_error + 1
            num_pre = num_pre + 1
        accurency = 1 - num_error/y_pre.shape[0]
    print(y_pre.shape[0], num_pre)
    print(accurency)

#read data
train_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\train\\'
test_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\test\\'    

df_train, seq_name= fa.read_csv_data(train_set_path)
df_test, seq_name= fa.read_csv_data(test_set_path)

print("train set shape: ", df_train.shape[0])
print("test set shape: ", df_test.shape[0])

#X_train = df_train.loc[:, ['w', 'qp', 'nvar', 'H', 'ngradx', 'ngrady', 'gmx', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]
#X_test = df_test.loc[:, ['w', 'qp','nvar', 'H',  'ngradx', 'ngrady', 'gmx', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]

""" X_train = df_train.loc[:, ['w', 'h', 'qp', 'nvar']]
X_test = df_test.loc[:, ['w', 'h', 'qp','nvar']]

y_train = df_train.loc[:, 'mode']
y_test = df_test.loc[:, 'mode']

y_train[y_train != 2000] = 1
y_train[y_train == 2000] = 0

y_test[y_test != 2000] = 1
y_test[y_test == 2000] = 0

scores = []
i = 0
for c in [10000]:
    scores.append([])
    for gamma in [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001,  0.0000000001]:
        svc = svm.SVC(kernel='rbf', C = c, gamma = gamma, probability = True)

        start_time = time.time()
        svc.fit(X_train, y_train)
        end_time = time.time()

        score = svc.score(X_test, y_test)
        print(c, gamma, score, svc.support_.shape[0], end_time-start_time)
        scores[i].append(score)

    i = i + 1 """


#y_hat=clf.predict(x_train)
#y_pre=clf.predict_proba(X_test)
# scores = pd.DataFrame(scores)
# print(scores)
# feature_train(df_train, df_test, ['w', 'qp', 'nvar', 'H', 'ngradx', 'ngrady', 'gmx', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
feature_train(df_train, df_test, ['qp', 'nvar', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])


