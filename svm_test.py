import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#read data
read_path = 'E:\\0-Research\\01-VVC\\result\\test'      
read_csv = glob.glob(os.path.join(read_path,'*.csv')) 
df = None

for i, path in enumerate(read_csv):     
    temp = pd.read_csv(path)
    temp = temp[temp['h'] == 32]
    temp = temp[temp['w'] == 32]
    #temp = temp[temp['w'] == temp['h']] 
    if df is None:          
        df = temp
    else:
        df = pd.concat([df, temp], ignore_index = True) 

X = df.loc[:, ['w', 'h', 'depth', 'qt_d', 'mt_d', 'qp', 'gradx', 'grady', 'var']]
Y = df.loc[:, 'mode']
Y[Y != 2000] = 1
Y[Y == 2000] = 0

clf = joblib.load('square.model')
print("测试集精度 %f" %clf.score(X, Y))   #测试集精度
y_pre=clf.predict_proba(X)

prelog = open('pre.csv', 'w')
for i in range(Y.shape[0]):
    if y_pre[i][0] < 0.5 and Y[i] == 1:
        iscorrect = 1
    if y_pre[i][0] < 0.5 and Y[i] == 0:
        iscorrect = 0
    if y_pre[i][0] > 0.5 and Y[i] == 1:
        iscorrect = 0
    if y_pre[i][0] > 0.5 and Y[i] == 0:
        iscorrect = 1
    prelog.write(str(Y[i]) + ',' + str(y_pre[i][0]) + ',' + str(y_pre[i][1]) +  ',' + str(iscorrect) + '\n')
prelog.close()   