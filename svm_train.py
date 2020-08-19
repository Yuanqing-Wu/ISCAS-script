import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import feature_analysis as fa

#read data
train_set_path = 'E:\\0-Research\\01-VVC\\result\\train\\'
test_set_path = 'E:\\0-Research\\01-VVC\\result\\test\\'    

df_train, seq_name= fa.read_csv_data(train_set_path)
df_test, seq_name= fa.read_csv_data(test_set_path)

df_train = df_train[df_train['w'] == 64]
df_train = df_train[df_train['h'] == 64]
df_test = df_test[df_test['w'] == 64]
df_test = df_test[df_test['h'] == 64]

X_train = df_train.loc[:, ['w', 'h', 'depth', 'qt_d', 'mt_d','qp', 'gradx', 'grady', 'var', 'ngz', 'nmg']]
X_test = df_test.loc[:, ['w', 'h', 'depth', 'qt_d', 'mt_d','qp', 'gradx', 'grady', 'var', 'ngz', 'nmg']]

Y_train = df_train.loc[:, 'mode']
Y_test = df_test.loc[:, 'mode']

Y_train[Y_train != 2000] = 1
Y_train[Y_train == 2000] = 0

Y_test[Y_test != 2000] = 1
Y_test[Y_test == 2000] = 0

 
tuned_parameters = {'C': [1000, 10000]}

svc = svm.SVC(kernel='rbf', probability = True)
clf = GridSearchCV(svc, tuned_parameters, scoring='accuracy')
clf.fit(X_train, Y_train)
#joblib.dump(clf, 'square.model')

print(clf.best_params_)
print(clf.best_score_)
#print("训练集精度 %f" %clf.score(x_train, y_train))  #训练集精度
print("测试集精度 %f" %clf.score(X_test, Y_test))   #测试集精度

#y_hat=clf.predict(x_train)
#y_pre=clf.predict_proba(X_test)
