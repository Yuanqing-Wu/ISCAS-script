import pandas as pd
import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import feature_analysis as fa

def feature_train(train, test, feature, train_size = 0):
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

    print(feature, svc.score(X_test, y_test))   



#read data
train_set_path = 'E:\\0-Research\\01-VVC\\result\\train\\'
test_set_path = 'E:\\0-Research\\01-VVC\\result\\test\\'    

#df_train, seq_name= fa.read_csv_data(train_set_path)
df_test, seq_name= fa.read_csv_data(test_set_path)

# df_train = df_train[df_train['w'] == df_train['h']]
# df_train_64x64 = df_train[df_train['w'] == 64]
# df_train_32x32 = df_train[df_train['w'] == 32]
# df_train_16x16 = df_train[df_train['w'] == 16]
# df_train_8x8 = df_train[df_train['w'] == 8]
# df_train_64x64 = fa.balance_set(df_train_64x64[df_train_64x64['mode'] == 2000], df_train_64x64[df_train_64x64['mode'] != 2000], 10000)
# df_train_32x32 = fa.balance_set(df_train_32x32[df_train_32x32['mode'] == 2000], df_train_32x32[df_train_32x32['mode'] != 2000], 10000)
# df_train_16x16 = fa.balance_set(df_train_16x16[df_train_16x16['mode'] == 2000], df_train_16x16[df_train_16x16['mode'] != 2000], 10000)
# df_train_8x8 = fa.balance_set(df_train_8x8[df_train_8x8['mode'] == 2000], df_train_8x8[df_train_8x8['mode'] != 2000], 10000)
# df_train_64x64.to_csv('s-ns_train_64x64_10000.csv')
# df_train_32x32.to_csv('s-ns_train_32x32_10000.csv')
# df_train_16x16.to_csv('s-ns_train_16x16_10000.csv')
# df_train_8x8.to_csv('s-ns_train_8x8_10000.csv')

#print("train set shape: ", df_train.shape[0])

df_test = df_test[df_test['w'] == df_test['h']]

df_test_64x64 = df_test[df_test['w'] == 64]
df_test_32x32 = df_test[df_test['w'] == 32]
df_test_16x16 = df_test[df_test['w'] == 16]
df_test_8x8 = df_test[df_test['w'] == 8]
df_test_64x64 = fa.balance_set(df_test_64x64[df_test_64x64['mode'] == 2000], df_test_64x64[df_test_64x64['mode'] != 2000], 20000)
df_test_32x32 = fa.balance_set(df_test_32x32[df_test_32x32['mode'] == 2000], df_test_32x32[df_test_32x32['mode'] != 2000], 20000)
df_test_16x16 = fa.balance_set(df_test_16x16[df_test_16x16['mode'] == 2000], df_test_16x16[df_test_16x16['mode'] != 2000], 20000)
df_test_8x8 = fa.balance_set(df_test_8x8[df_test_8x8['mode'] == 2000], df_test_8x8[df_test_8x8['mode'] != 2000], 20000)
df_test_64x64.to_csv('s-ns_test_64x64_20000.csv')
df_test_32x32.to_csv('s-ns_test_32x32_20000.csv')
df_test_16x16.to_csv('s-ns_test_16x16_20000.csv')
df_test_8x8.to_csv('s-ns_test_8x8_20000.csv')

print("test set shape: ", df_test.shape[0])

# X_train = df_train.loc[:, ['w', 'qp', 'var']]
# X_test = df_test.loc[:, ['w', 'qp', 'var']]

# y_train = df_train.loc[:, 'mode']
# y_test = df_test.loc[:, 'mode']

# y_train[y_train != 2000] = 1
# y_train[y_train == 2000] = 0

# y_test[y_test != 2000] = 1
# y_test[y_test == 2000] = 0

# X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size = 0.01, random_state = 1, stratify = y_train)
 
#tuned_parameters = {'C': [1000, 10000]}

# svc = svm.SVC(kernel='rbf', C = 10000, probability = True)
# svc.fit(X_train, y_train)

#clf = GridSearchCV(svc, tuned_parameters, scoring='accuracy')
#clf.fit(X_train, Y_train)
# joblib.dump(svc, 's_ns_square.model')

#print(clf.best_params_)
#print(clf.best_score_)
#print("训练集精度 %f" %clf.score(x_train, y_train))  #训练集精度
# print("测试集精度 %f" %svc.score(X_test, y_test))   #测试集精度

#y_hat=clf.predict(x_train)
#y_pre=clf.predict_proba(X_test)

start_time = time.time()
feature_train(df_train, df_test, ['w', 'qp', 'H', 'nvar', 'ngradx', 'ngrady', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'], 0.1)
end_time = time.time()
print("time:%d"  % (end_time-start_time)) 