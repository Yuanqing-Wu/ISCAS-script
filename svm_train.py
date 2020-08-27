import pandas as pd
import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import feature_analysis as fa

def feature_train(train, test, feature, train_size = 0, save_mode = None):

    start_time = time.time()

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

    svc = svm.SVC(kernel='rbf', C = 10000, probability = True)
    svc.fit(X_train, y_train)

    if save_mode != None:
        joblib.dump(svc, save_mode + '.model')
        

    print(feature, svc.score(X_test, y_test))   

    end_time = time.time()
    print("time:%d"  % (end_time-start_time)) 
    print()

    



#read data
train_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\train\\'
test_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\test\\'    

df_train, seq_name= fa.read_csv_data(train_set_path)
df_test, seq_name= fa.read_csv_data(test_set_path)

print("train set shape: ", df_train.shape[0])
print("test set shape: ", df_test.shape[0])

X_train = df_train.loc[:, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]
X_test = df_test.loc[:, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]

y_train = df_train.loc[:, 'mode']
y_test = df_test.loc[:, 'mode']

y_train[y_train != 2000] = 1
y_train[y_train == 2000] = 0

y_test[y_test != 2000] = 1
y_test[y_test == 2000] = 0

#print()

#print(df_train['qp'].value_counts(1))
 
tuned_parameters = {'C': [1000, 10000,50000]}

svc = svm.SVC(kernel='rbf', probability = True)
# svc.fit(X_train, y_train)
start_time = time.time()

clf = GridSearchCV(svc, tuned_parameters, scoring='accuracy')
clf.fit(X_train, y_train)
end_time = time.time()
print("time:%d"  % (end_time-start_time)) 
print()

cv_results = clf.cv_results_
print(cv_results)
cv_results.to_csv('cv_results.csv')
print()
# joblib.dump(svc, 's_ns_square.model')

#print(clf.best_params_)
#print(clf.best_score_)
#print("训练集精度 %f" %clf.score(x_train, y_train))  #训练集精度
# print("测试集精度 %f" %svc.score(X_test, y_test))   #测试集精度

#y_hat=clf.predict(x_train)
#y_pre=clf.predict_proba(X_test)


#feature_train(df_train, df_test, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'], save_mode='all_64x64_32x32_16x_16')


# df_test64_64 = df_test[df_test['w'] == 64]
# df_test32_32 = df_test[df_test['w'] == 32]
# df_test16_16 = df_test[df_test['w'] == 16]

# X64_64 = df_test64_64.loc[:, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]
# X32_32 = df_test32_32.loc[:, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]
# X16_16 = df_test16_16.loc[:, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]


# y64_64 = df_test64_64.loc[:, 'mode']
# y32_32 = df_test32_32.loc[:, 'mode']
# y16_16 = df_test16_16.loc[:, 'mode']


# y64_64[y64_64!= 2000]=1
# y64_64[y64_64== 2000]=0
# y32_32[y32_32!= 2000]=1
# y32_32[y32_32== 2000]=0
# y16_16[y16_16!= 2000]=1
# y16_16[y16_16== 2000]=0


# svc = joblib.load('all_64x64_32x32_16x_16.model')

# y_pre=svc.predict_proba(X64_64)
# print('64_64', svc.score(X64_64, y64_64))
# print('32_32', svc.score(X32_32, y32_32))
# print('16_16', svc.score(X16_16, y16_16))
