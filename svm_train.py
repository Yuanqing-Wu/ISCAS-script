#2020-8-12
#wgq

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
#from sklearn.neural_network import MLPClassifier

import minepy as mp

#read data
read_path = 'E:\\0-Research\\01-VVC\\result\\8_18'      
read_csv = glob.glob(os.path.join(read_path,'*.csv')) 
df = None

for i, path in enumerate(read_csv):     
    temp = pd.read_csv(path)
    #temp = temp[temp['h'] == 32]
    #temp = temp[temp['w'] == 32]
    temp = temp[temp['w'] == temp['h']] 
    if df is None:          
        df = temp
    else:
        df = pd.concat([df, temp], ignore_index = True) 

#print(df["mode"].value_counts(1) )
# balance data
#df = df.loc[1:5000, :]

s_no = df[df['mode'] == 2000]  
s_sp = df[df['mode'] != 2000]
if s_no.shape[0] < s_sp.shape[0]:
    s_sp = s_sp.sample(n=int(len(s_no)), replace=False, random_state=0, axis=0)
if s_no.shape[0] > s_sp.shape[0]:
    s_no = s_no.sample(n=int(len(s_sp)), replace=False, random_state=0, axis=0)
df = pd.concat([s_sp, s_no], axis=0)
#a = df["mode"].value_counts(1)

#process data
#X = df.loc[:, ['w', 'h', 'depth', 'qt_d', 'mt_d','qp', 'gradx', 'grady', 'var']]
#X = df.loc[:, ['gradx', 'grady', 'var', 'ngz', 'nmg', 'ubd', 'lrd']]
X = df.loc[:, ['w', 'h', 'depth', 'qt_d', 'mt_d', 'qp', 'gradx', 'grady', 'var']]
A1 = df[df['mode'] == 2000]
A2 = df[df['mode'] != 2000]

#plt.figure(num='gradx')
#B1 = A1.loc[:, ['gradx']]
#B2 = A2.loc[:, ['gradx']]
#plt.scatter(range(0, B1.shape[0]), B1, c = 'r', marker = 'o')
#plt.scatter(range(0, B1.shape[0]), B2, c = 'b', marker = 'o')
#ax = B1.plot(, y = B1, kind='scatter', ylim = [-10000, 40000])
#B2.plot(x=range(1, B2.shape[0]), y = B2, kind='scatter', ylim = [-10000, 40000], ax=ax)
#plt.show()

plt.figure(num='var')
B1 = A1.loc[:, ['var']]
B2 = A2.loc[:, ['var']]
#plt.scatter(range(0, B1.shape[0]), B1, c = 'r', marker = 'o')
#plt.scatter(range(0, B1.shape[0]), B2, c = 'b', marker = 'o')
ax = B1.plot(kind='kde')
B2.plot(kind='kde', ax=ax, xlim = [-200, 2000])
plt.show()
#X['r_g'] = r_g
Y = df.loc[:, 'mode']
Y[Y != 2000] = 1
Y[Y == 2000] = 0
#print(Y)
print(X.info())
print(Y.value_counts(1))

# mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
# mine.compute_score(np.array(df.loc[:, 'var']), np.array(df.loc[:, 'mode']))
# print("var", mine.mic())
# mine.compute_score(np.array(df.loc[:, 'gradx']), np.array(df.loc[:, 'mode']))
# print("gradx", mine.mic())
# mine.compute_score(np.array(df.loc[:, 'grady']), np.array(df.loc[:, 'mode']))
# print("grady", mine.mic())
# mine.compute_score(np.array(df.loc[:, 'qp']), np.array(df.loc[:, 'mode']))
# print("qp", mine.mic())
# mine.compute_score(np.array(df.loc[:, 'ngz']), np.array(df.loc[:, 'mode']))
# print("ngz", mine.mic())
# mine.compute_score(np.array(df.loc[:, 'nmg']), np.array(df.loc[:, 'mode']))
# print("nmg", mine.mic())

#split data set
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.8, random_state = 1)  

C = 20000
gamma = 1/X.shape[1]
# fit the model
#clf = svm.SVC(C=C, kernel='rbf', probability = True)
#clf = MLPClassifier(hidden_layer_sizes=(6,5), max_iter=200, alpha=1e-4, batch_size =200, random_state=1, learning_rate_init=.001)
#clf.fit(x_train, y_train)
#print(clf.coef_)
#joblib.dump(clf, 'square.model')
clf = joblib.load('square.model')

print("训练集精度 %f" %clf.score(x_train, y_train))  #训练集精度
print("测试集精度 %f" %clf.score(x_test, y_test))   #测试集精度

y_hat=clf.predict(x_train)
#y_pre=clf.predict(x_test)
y_pre=clf.predict_proba(x_test)
#fmse=(y_pre-y_test).T.dot(y_pre-y_test)/len(y_test)   #测试集均方误差
#mse=(y_hat-y_train).T.dot(y_hat-y_train)/len(y_train)  #训练集

 
#print(y_test)
#print(y_pre)
