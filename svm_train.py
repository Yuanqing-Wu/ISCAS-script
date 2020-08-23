import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import feature_analysis as fa

def one_feature_train(train, test, feature):
    X_train = train.loc[:, [feature]]
    X_test = test.loc[:, [feature]]

    Y_train = train.loc[:, 'mode']
    Y_test = test.loc[:, 'mode']

    Y_train = Y_train.copy()
    Y_test = Y_test.copy()

    Y_train[Y_train != 2000] = 1
    Y_train[Y_train == 2000] = 0
    print("train set: ", Y_train.value_counts())

    Y_test[Y_test != 2000] = 1
    Y_test[Y_test == 2000] = 0
    print("test set: ", Y_test.value_counts())

    svc = svm.SVC(kernel='rbf', C = 10000, probability = True)
    svc.fit(X_train, Y_train)

    print(feature + ': ', svc.score(X_test, Y_test))   

#read data
train_set_path = 'E:\\0-Research\\01-VVC\\result\\train\\'
test_set_path = 'E:\\0-Research\\01-VVC\\result\\test\\'    

df_train, seq_name= fa.read_csv_data(train_set_path)
df_test, seq_name= fa.read_csv_data(test_set_path)

df_train = df_train[df_train['w'] == 32]
df_train = df_train[df_train['h'] == 32]
df_train = df_train[df_train['qp'] == 32]
df_train = fa.balance_set(df_train[df_train['mode'] == 2000], df_train[df_train['mode'] != 2000])
print("train set shape: ", df_train.shape[0])

df_test = df_test[df_test['w'] == 32]
df_test = df_test[df_test['h'] == 32]
df_test = df_test[df_test['qp'] == 32]
df_test = fa.balance_set(df_test[df_test['mode'] == 2000], df_test[df_test['mode'] != 2000])
print("test set shape: ", df_test.shape[0])

# X_train = df_train.loc[:, ['H']]
# X_test = df_test.loc[:, ['H']]

# Y_train = df_train.loc[:, 'mode']
# Y_test = df_test.loc[:, 'mode']

# Y_train[Y_train != 2000] = 1
# Y_train[Y_train == 2000] = 0

# Y_test[Y_test != 2000] = 1
# Y_test[Y_test == 2000] = 0

 
#tuned_parameters = {'C': [1000, 10000]}

# svc = svm.SVC(kernel='rbf', C = 10000, probability = True)
# svc.fit(X_train, Y_train)

#clf = GridSearchCV(svc, tuned_parameters, scoring='accuracy')
#clf.fit(X_train, Y_train)
#joblib.dump(clf, 'square.model')

#print(clf.best_params_)
#print(clf.best_score_)
#print("训练集精度 %f" %clf.score(x_train, y_train))  #训练集精度
# print("测试集精度 %f" %svc.score(X_test, Y_test))   #测试集精度

#y_hat=clf.predict(x_train)
#y_pre=clf.predict_proba(X_test)

one_feature_train(df_train, df_test, 'var')
one_feature_train(df_train, df_test, 'H')
one_feature_train(df_train, df_test, 'gradx')
one_feature_train(df_train, df_test, 'grady')
one_feature_train(df_train, df_test, 'maxgrad')
one_feature_train(df_train, df_test, 'dvarh')
one_feature_train(df_train, df_test, 'dvarv')
one_feature_train(df_train, df_test, 'dHh')
one_feature_train(df_train, df_test, 'dHv')
one_feature_train(df_train, df_test, 'dgradxh')
one_feature_train(df_train, df_test, 'dgradxv')
one_feature_train(df_train, df_test, 'dgradyh')
one_feature_train(df_train, df_test, 'dgradyv')
