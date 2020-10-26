import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import subprocess


def stand_sca(data):
    """
    标准差标准化
    :param data:传入的数据
    :return:标准化之后的数据
    """
    new_data=(data-data.mean())/data.std()   
    return new_data

'''
@写libsvm数据
    write_data_sns(classifier, df, feature)
    write_subdata_sns(classifier, df, feature)
    write_data_hsvs(classifier, df, feature)
    write_subdata_hsvs(classifier, df, feature)
    writesns(classifier, feature)
    write2hsvs(classifier, feature)
'''

def write_data_sns(classifier, df, feature):

    if not os.path.exists('libsvmdata/' + classifier):
        os.mkdir('libsvmdata/' + classifier)

    data_file = open('libsvmdata/'  + classifier + '/test' + '.txt', 'w')

    y = df.loc[:, 'mode']
    X = df.loc[:, feature]
    X = stand_sca(X) #标准化

    for i in range(df.shape[0]):
        label = y[i]
        if(label == 2000):
            label = 0
        else:
            label = 1
        data_file.write(str(label) + ' ')
        n = 0
        for m in feature:
            data_file.write(str(n+1) + ':' + str(X.loc[i, m]) + ' ')
            n = n + 1
        data_file.write('\n')
    data_file.close()

def write_subdata_sns(classifier, df, feature):
    X = df.loc[:, feature]
    y = df.loc[:, 'mode']
    X = stand_sca(X) #标准化  

    if not os.path.exists('libsvmdata/' + classifier):
        os.mkdir('libsvmdata/' + classifier)
    for i in range(20):
        X_train, _, y_train, _ = train_test_split(X, y, train_size=500, stratify=y)      
        data_file = open('libsvmdata/'  + classifier + '/train' + str(i) + '.txt', 'w')
        for m in X_train.index:
            label = y_train[m]
            if(label == 2000):
                label = 0
            else:
                label = 1
            data_file.write(str(label) + ' ')
            n = 0
            for k in feature:
                data_file.write(str(n+1) + ':' + str(X_train.loc[m, k]) + ' ')
                n = n + 1
            data_file.write('\n')
        data_file.close()

def write_data_hsvs(classifier, df, feature):

    if not os.path.exists('libsvmdata/' + classifier):
        os.mkdir('libsvmdata/' + classifier)

    data_file = open('libsvmdata/'  + classifier + '/test' + '.txt', 'w')

    for i in range(df.shape[0]):
        data_file.write(str(df.loc[i, 'mode']) + ' ')
        n = 0
        for m in feature:
            data_file.write(str(n+1) + ':' + str(df.loc[i, m]) + ' ')
            n = n + 1
        data_file.write('\n')
    data_file.close()

def write_subdata_hsvs(classifier, df, feature):
    X = df.loc[:, feature]
    y = df.loc[:, 'mode']   
    if not os.path.exists('libsvmdata/' + classifier):
        os.mkdir('libsvmdata/' + classifier)
    for i in range(20):
        X_train, _, y_train, _ = train_test_split(X, y, train_size=100, stratify=y)      
        data_file = open('libsvmdata/'  + classifier + '/train' + str(i) + '.txt', 'w')
        for m in X_train.index:
            data_file.write(str(y_train[m]) + ' ')
            n = 0
            for k in feature:
                data_file.write(str(n+1) + ':' + str(X_train.loc[m, k]) + ' ')
                n = n + 1
            data_file.write('\n')
        data_file.close()

def writesns(classifier, feature):
    path = "csv_data/" + classifier + "_0.csv"
    df = pd.read_csv(path)
    write_subdata_sns(classifier, df, feature)
    write_data_sns(classifier, df, feature)

def write2hsvs(classifier, feature):
    path = "csv_data/" + classifier + "_0.csv"
    df = pd.read_csv(path)
    #write_subdata_hsvs(classifier, df, feature)
    write_data_hsvs(classifier, df, feature)

def cal_mean_std(classifier, feature):
    print(classifier)
    path = "csv_data/" + classifier + "_0.csv"
    df = pd.read_csv(path)
    print("mean: ")
    print(df[feature].mean())
    print()
    print("std: ")
    print(df[feature].std())

'''
Run libsvm
    run_ParaSearch(exe, train_file, test_file, model_file, predict_file, log_file)
    run_th(exe, test_file, model_file, predict_file, log_file)
    train_sub_set(exe, classifier)
    select_th(exe, classifier)
'''

def run_predict(exe, test_file, model_file, predict_file, log_file):

    cmd = exe + ' -b 1 ' + test_file + ' ' + model_file + ' ' + predict_file + ' > ' + log_file      
    subprocess.Popen(cmd, shell=True)

def run_ParaSearch(exe, train_file, test_file, model_file, predict_file, log_file):

    cmd = exe + ' -i ' + train_file + ' -e ' + test_file + ' -m ' + model_file + ' -p ' + predict_file + ' > ' + log_file      
    subprocess.Popen(cmd, shell=True)

def run_th(exe, test_file, model_file, predict_file, log_file):

    cmd = exe + ' -e ' + test_file + ' -m ' + model_file + ' -p ' + predict_file + ' > ' + log_file
    subprocess.Popen(cmd, shell=True)

def train_sub_set(exe, classifier):

    path = 'libsvmdata/' + classifier + '/'
    log_path = path + 'log/'
    
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    
    test_file = path + 'test.txt'
    for i in range(20):
        train_file = path + 'train' + str(i) + '.txt'
        model_file = log_path + 'model' + str(i) + '.txt'
        predict_file = log_path + 'predict' + str(i) + '.txt'
        log_file = log_path + 'log' + str(i) + '.txt'
        run_ParaSearch(exe, train_file, test_file, model_file, predict_file, log_file)

def select_th(exe, classifier):

    path = 'libsvmdata/' + classifier + '/'
    log_path = path + 'log/'

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    
    test_file = path + 'test.txt'
    model_file = log_path + 'model' + '.txt'
    predict_file = log_path + 'predict' + '.txt'
    log_file = log_path + 'log' + '.txt'
    run_th(exe, test_file, model_file, predict_file, log_file)

def read_log(logs_path):

    classifer_path = os.listdir(logs_path)
    row = 1
    for classifer in classifer_path:
        print(classifer)
        logs = os.listdir(logs_path + classifer+'/log/')
        for log_name in logs:
            if log_name.startswith('log'):
                _, num = log_name.split('log')
                num, _ = num.split('.txt')
                file = open(logs_path + classifer + '/log/' + log_name, 'r')
                line = ' '
                while not line.startswith('Summary:'):
                    line = file.readline()
                line = file.readline()
                line = line.strip()
                print(num, line)
        print()
# 写libsvm data
if 0:
    writesns('s-ns_64x64', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_16x16', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_8x8', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_32x16', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_32x8', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_32x4', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_16x8', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_16x4', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    writesns('s-ns_8x4', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])

# 计算均值和方差
if 0:
    cal_mean_std('s-ns_64x64', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_32x32', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_16x16', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_8x8', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_32x16', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_32x8', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_32x4', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_16x8', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_16x4', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    cal_mean_std('s-ns_8x4', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])


# Run ParaSearch
if 0:
    ParaSearch = 'libsvmexe/ParaSearch'

# Run ParaSearch 子集
if 0:
    train = "libsvmexe/train"
    predict = "libsvmexe/predict"
    ParaSearch = 'libsvmexe/ParaSearch'
    ParaSearchs = 'libsvmexe/ParaS'
    th = 'libsvmexe/th'

    train_sub_set(ParaSearchs, 's-ns_64x64')
    train_sub_set(ParaSearchs, 's-ns_32x32')
    train_sub_set(ParaSearchs, 's-ns_16x16')
    train_sub_set(ParaSearchs, 's-ns_8x8')
    train_sub_set(ParaSearchs, 's-ns_32x16')
    train_sub_set(ParaSearchs, 's-ns_32x8')
    train_sub_set(ParaSearchs, 's-ns_32x4')
    train_sub_set(ParaSearchs, 's-ns_16x8')
    train_sub_set(ParaSearchs, 's-ns_16x4')
    train_sub_set(ParaSearchs, 's-ns_8x4')

# Run predict
if 0:
    predict = "libsvmexe/predict"
    for i in range(20):
        run_predict(predict, 'libsvmexe/test.txt', 'libsvmdata/s-ns_64x64/log/model' + str(i) + '.txt', 'libsvmexe/predict' + str(i) + '.txt', 'libsvmexe/log.txt')

# Run log
if 1:
    read_log('libsvmdata/')
