import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import subprocess
import minepy as mp
import argparse
import shutil
import time
import matplotlib.pyplot as plt


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

    X, _, y, _ = train_test_split(X, y, train_size=10000, stratify=y)

    for i in X.index:
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
    for i in range(30):
        X_train, _, y_train, _ = train_test_split(X, y, train_size=200, stratify=y)      
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
    
    y = df.loc[:, 'mode']
    X = df.loc[:, feature]
    X = stand_sca(X) #标准化
    X, _, y, _ = train_test_split(X, y, train_size=10000, stratify=y)
    data_file = open('libsvmdata/'  + classifier + '/test' + '.txt', 'w')

    for i in X.index:
        data_file.write(str(df.loc[i, 'mode']) + ' ')
        n = 0
        for m in feature:
            data_file.write(str(n+1) + ':' + str(X.loc[i, m]) + ' ')
            n = n + 1
        data_file.write('\n')
    data_file.close()

def write_subdata_hsvs(classifier, df, feature):
    X = df.loc[:, feature]
    y = df.loc[:, 'mode']   
    X = stand_sca(X) #标准化  

    if not os.path.exists('libsvmdata/' + classifier):
        os.mkdir('libsvmdata/' + classifier)
    for i in range(30):
        X_train, _, y_train, _ = train_test_split(X, y, train_size=200, stratify=y)      
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
    df['ngrad'] = df.loc[:, 'ngradx'] + df.loc[:, 'ngrady']
    df['MaxDiffgrad'] = df.loc[:, 'MaxDiffgradx'] + df.loc[:, 'MaxDiffgrady']
    df['Inconsgradh'] = df.loc[:, 'Inconsgradxh'] + df.loc[:, 'Inconsgradyh']
    df['Inconsgradv'] = df.loc[:, 'Inconsgradxv'] + df.loc[:, 'Inconsgradyv']
    write_subdata_sns(classifier, df, feature)
    write_data_sns(classifier, df, feature)

def writehsvs(classifier, feature):
    path = "csv_data/" + classifier + "_0.csv"
    df = pd.read_csv(path)
    write_subdata_hsvs(classifier, df, feature)
    write_data_hsvs(classifier, df, feature)

def cal_mean_std_sns(classifier, feature):
    print(classifier)
    path = "csv_data/" + classifier + "_0.csv"
    df = pd.read_csv(path)
    mea = np.array(df[feature].mean())
    st = np.array(df[feature].std())
    print(classifier + 'mean = {', mea[0], ',', mea[1], ',', mea[2],',', mea[3],',', mea[4],',', mea[5],',', mea[6],',', mea[7],'}')
    print(classifier + 'std = {', st[0], ',', st[1], ',', st[2],',', st[3],',', st[4],',', st[5],',', st[6],',', st[7],'}')

def cal_mean_std_hsvs(classifier, feature):
    print(classifier)
    path = "csv_data/" + classifier + "_0.csv"
    df = pd.read_csv(path)
    mea = np.array(df[feature].mean())
    st = np.array(df[feature].std())
    print(classifier + 'mean = {', mea[0], ',', mea[1], ',', mea[2],',', mea[3],',', mea[4],',', mea[5], '}')
    print(classifier + 'std = {', st[0], ',', st[1], ',', st[2],',', st[3],',', st[4],',', st[5],'}')

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
    for i in range(30):
        train_file = path + 'train' + str(i) + '.txt'
        model_file = log_path + 'model' + str(i) + '.txt'
        predict_file = log_path + 'predict' + str(i) + '.txt'
        log_file = log_path + 'log' + str(i) + '.txt'
        run_ParaSearch(exe, train_file, test_file, model_file, predict_file, log_file)

def select_th(exe, classifier):

    path = 'libsvmdata/' + classifier + '/'
    log_path = path + 'log/'
    model_path = 'model/'

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    
    test_file = path + 'test.txt'
    model_file = model_path + classifier + '.txt'
    predict_file = log_path + 'predict' + '.txt'
    log_file = log_path + 'log' + '.txt'
    run_th(exe, test_file, model_file, predict_file, log_file)

def read_log(logs_path):

    output_path = 'model/'
    classifer_path = os.listdir(logs_path)
    row = 1
    for classifer in classifer_path:
        #print(classifer)
        logs = os.listdir(logs_path + classifer+'/log/')
        max_acc = 0
        max_seq = 0
        for log_name in logs:
            if log_name.startswith('log') and not log_name.startswith('log.txt'):
                _, num = log_name.split('log')
                num, _ = num.split('.txt')
                file = open(logs_path + classifer + '/log/' + log_name, 'r')
                line = ' '
                while not line.startswith('Summary:'):
                    line = file.readline()
                line = file.readline()
                line = line.strip()
                before, _ = line.split('%')
                _, acc = before.split('=')
                acc = acc.strip()
                if float(acc) > max_acc:
                    max_acc =  float(acc)
                    max_seq = num
        print(classifer + ': ', max_acc, max_seq)
        shutil.copyfile(logs_path + classifer + '/log/model' + str(max_seq) + '.txt', output_path + classifer + '.txt')


def readth_log(logs_path):

    classifer_path = os.listdir(logs_path)
    row = 1
    df = pd.DataFrame()   
    for classifer in classifer_path:
        #print(classifer)
        logs = os.listdir(logs_path + classifer+'/log/')
        th = []
        acc = []
        for log_name in logs:          
            if log_name.startswith('log.txt'):
                file = open(logs_path + classifer + '/log/' + log_name, 'r')
                line = ' '
                while not line.startswith('th = 1'):
                    if line.startswith('th'):
                        line = line.strip()
                        before, _ = line.split('%')
                        before, ac = before.split('Accuracy = ')
                        _, t = before.split('= ')
                        acc.append(float(ac.strip()))
                        th.append(float(t.strip()))
                        #print(line)
                    line = file.readline()
                #print(line)
        df['th'] = th
        df[classifer] = acc    
    #print(df)    
    #plt.figure(num='th')  
    fig, ax = plt.subplots()
    #ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_64x64'], label = '64x64', linewidth=2.0)
    ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_32x32'], label = '32x32', linewidth=2.0)
    ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_16x16'], label = '16x16', linewidth=2.0)
    #ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_8x8'], label = '8x8', linewidth=2.0)
    ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_32x16'], label = '32x16', linewidth=2.0)
    ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_32x8'], label = '32x8', linewidth=2.0)
    #ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_32x4'], label = '32x4', linewidth=2.0)
    ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_16x8'], label = '16x8', linewidth=2.0)
    #ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_16x4'], label = '16x4', linewidth=2.0)
    #ax.plot(df.loc[:, 'th'], df.loc[:, 's-ns_8x4'], label = '8x4', linewidth=2.0)

    #df.plot.line(x= 'th', y='s-ns_64x64') 
    ax.legend(fontsize=15)

    plt.xlim((0.5, 1.01))
    plt.ylim((70, 101))
    #设置坐标轴名称
    plt.xlabel('Threshold',fontsize=15)
    plt.ylabel('Accuracy (%)',fontsize=15)
  
    #设置坐标轴刻度
    my_x_ticks = np.arange(0.5, 1.01, 0.1)
    my_y_ticks = np.arange(70, 101, 5)
    plt.xticks(my_x_ticks,fontsize=15)
    plt.yticks(my_y_ticks,fontsize=15)
    plt.grid()
    plt.savefig(fname = 'snsth.png', dpi=500)
    #print(df)    

    fig, ax2 = plt.subplots()
    ax2.plot(df.loc[:, 'th'], df.loc[:, 'hs-vs_32x32'], label = '32x32', linewidth=2.0)
    ax2.plot(df.loc[:, 'th'], df.loc[:, 'hs-vs_16x16'], label = '16x16', linewidth=2.0)
    ax2.plot(df.loc[:, 'th'], df.loc[:, 'hs-vs_32x16'], label = '32x16', linewidth=2.0)
    ax2.plot(df.loc[:, 'th'], df.loc[:, 'hs-vs_32x8'], label = '32x8', linewidth=2.0)
    ax2.plot(df.loc[:, 'th'], df.loc[:, 'hs-vs_16x8'], label = '16x8', linewidth=2.0)
    #ax2.plot(df.loc[:, 'th'], df.loc[:, 'hs-vs_8x8'], label = '8x8', linewidth=2.0)

    #df.plot.line(x= 'th', y='s-ns_64x64') 
    ax2.legend(fontsize=15)

    plt.xlim((0.5, 1.01))
    plt.ylim((65, 101))
    #设置坐标轴名称
    plt.xlabel('Threshold', fontsize=15)
    plt.ylabel('Accuracy (%)', fontsize=15)
  
    #设置坐标轴刻度
    my_x_ticks = np.arange(0.5, 1.01, 0.1)
    my_y_ticks = np.arange(65, 101, 5)
    plt.xticks(my_x_ticks, fontsize=15)
    plt.yticks(my_y_ticks, fontsize=15)
    plt.grid()
    plt.savefig(fname = 'hsvsth.png', dpi=500)

def read_csv_data(read_path):

    read_csv = glob.glob(os.path.join(read_path,'*.csv')) # read the address of every csv file

    df = None
    seq_name = []

    for i, path in enumerate(read_csv):     #  Loop reading every csv file

        before, file_name = path.split(read_path)
        file_name, csv = file_name.split('.')
        seq_name.append(file_name)

        month = pd.read_csv(path)
        if df is None:          # if df is empty
            df = month
        else:
            df = pd.concat([df,month], ignore_index = True)  # emerge every csv data
    
    return df, seq_name

def balance_set(set1, set2, size = 0):

    # balance trainning data set 
    if size > set1.shape[0] or size > set2.shape[0]:
        print("the size is too large")
        return 

    if size == 0:
        if set1.shape[0] < set2.shape[0]:
            set2 = set2.sample(n=int(len(set1)), replace=False, axis=0)
        if set1.shape[0] > set2.shape[0]:
            set1 = set1.sample(n=int(len(set2)), replace=False, axis=0)
    else:
        set1 = set1.sample(n=size, replace=False, axis=0)
        set2 = set2.sample(n=size, replace=False, axis=0)
    df = pd.concat([set2, set1], axis=0)

    return df

def save_block_set_sns(df, w, h, file_name, data_size = 0, isbalance_set = True):
    df = df[df['w'] == w]
    df = df[df['h'] == h]

    if isbalance_set:
        if data_size == 0:
            df = balance_set(df[df['mode'] == 2000], df[df['mode'] != 2000])
        else:
            df = balance_set(df[df['mode'] == 2000], df[df['mode'] != 2000], data_size)
    file_name = file_name + '_' + str(w) + 'x' +str(h) + '_' +str(data_size) + '.csv'
    df.to_csv('csv_data/' + file_name)

def save_block_set_hsvs(df, w, h, file_name, data_size = 0, isbalance_set = True):
    df = df[df['w'] == w]
    df = df[df['h'] == h]

    for i in df.index:
        if df.loc[i, 'mode'] == 2 or df.loc[i, 'mode'] == 4:
            df.loc[i, 'mode'] = 100  # HS
        if df.loc[i, 'mode'] == 3 or df.loc[i, 'mode'] == 5:
            df.loc[i, 'mode'] = 200  # VS
    
    if isbalance_set:
        if data_size == 0:
            df = balance_set(df[df['mode'] == 100], df[df['mode'] == 200])
        else:
            df = balance_set(df[df['mode'] == 100], df[df['mode'] == 200], data_size)
    file_name = file_name + '_' + str(w) + 'x' +str(h) + '_' +str(data_size) + '.csv'
    df.to_csv('csv_data/' + file_name)

def mic(x, y):

    # calculate the maximal information coefficient 

    x = np.array(x)
    y = np.array(y)
    mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)

    return mine.mic()

def feature_train(classifier, feature):

    if classifier.startswith('s'):
        writesns(classifier, feature)
    else:
        writehsvs(classifier, feature)
    print(feature)

    ParaSearchs = 'libsvmexe/ParaSs'
    train_sub_set(ParaSearchs, classifier)
    time.sleep(5)
    while 1:
        file_handle = os.popen('ps -a | grep ' + 'ParaSs' + '| wc -l')
        Numproc = int(file_handle.read())
        if Numproc == 0:
            break
        time.sleep(1)
    #read_log('libsvmdata/') 
###################################################################################################################################

# 保存块data sns
if 0:
    read_path = 'train/'      # the path of csv file
    df, seq_name = read_csv_data(read_path)

    df = df[df['chType'] == 0]

    save_block_set_sns(df, 64, 64,'s-ns')
    save_block_set_sns(df, 32, 32,'s-ns')
    save_block_set_sns(df, 16, 16,'s-ns')
    save_block_set_sns(df, 8, 8,'s-ns')
    save_block_set_sns(df, 32, 16,'s-ns')
    save_block_set_sns(df, 32, 8,'s-ns')
    save_block_set_sns(df, 32, 4,'s-ns')
    save_block_set_sns(df, 16, 8,'s-ns')
    save_block_set_sns(df, 16, 4,'s-ns')
    save_block_set_sns(df, 8, 4,'s-ns')

# 保存块data hsvs
if 0:
    read_path = 'train/'      # the path of csv file
    df, seq_name = read_csv_data(read_path)

    df = df[df['chType'] == 0]

    save_block_set_hsvs(df, 32, 32,'hs-vs')
    save_block_set_hsvs(df, 16, 16,'hs-vs')
    save_block_set_hsvs(df, 8, 8,'hs-vs')
    save_block_set_hsvs(df, 32, 16,'hs-vs')
    save_block_set_hsvs(df, 32, 8,'hs-vs')
    save_block_set_hsvs(df, 16, 8,'hs-vs')

# 计算 mic
if 0:
    train_set_path = 'csv_data/hs-vs_16x8_0.csv'  
    print(train_set_path)  
    df = pd.read_csv(train_set_path)

    y = df.loc[:, 'mode']
    
    
    feature = ['qp', 'var', 'ndvarh','ndvarv','ndva','MaxDiffVar','InconsVarH','InconsVarV','ngradx','ngrady','ndgradxh','ndgradxv','ndgradyh','ndgradyv','ndgradx','ndgrady','gmx']
    for f in feature:
        print(f + ": ", mic(y, df.loc[:, f]))    

# 计算 mic 标准化
if 0:
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--classifier", default='s-ns_64x64')
    args = parser.parse_args()
    classifier = args.classifier
    train_set_path = 'csv_data/' + classifier + '_0.csv'  
    print(train_set_path)  
    df = pd.read_csv(train_set_path)

    y = df.loc[:, 'mode']
    if classifier.startswith('s-ns'):
        y[y != 2000] = 1
        y[y == 2000] = 0
    df['ngrad'] = df.loc[:, 'ngradx'] + df.loc[:, 'ngrady']
    df['MaxDiffgrad'] = df.loc[:, 'MaxDiffgradx'] + df.loc[:, 'MaxDiffgrady']
    df['Inconsgradh'] = df.loc[:, 'Inconsgradxh'] + df.loc[:, 'Inconsgradyh']
    df['Inconsgradv'] = df.loc[:, 'Inconsgradxv'] + df.loc[:, 'Inconsgradyv']
    df = stand_sca(df)
    
    feature = ['qp', 'var','ngradx','ngrady','gmx','ndva', 'ndgradx','ndgrady','MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady']
    for f in feature:
        print(f + ": ", mic(y, df.loc[:, f]))   

# 写libsvm data sns
if 0:
    # writesns('s-ns_64x64', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    writesns('s-ns_32x32', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # writesns('s-ns_16x16', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    # # writesns('s-ns_8x8', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    # writesns('s-ns_32x16', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    # writesns('s-ns_32x8', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    # # writesns('s-ns_32x4', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    # writesns('s-ns_16x8', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    # writesns('s-ns_16x4', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])
    # writesns('s-ns_8x4', ['qp', 'var', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx', 'ngrad', 'MaxDiffgrad', 'Inconsgradh', 'Inconsgradv'])

# 写libsvm data hsvs
if 0:
    writehsvs('hs-vs_32x32', ['qp', 'ndva', 'ngradx', 'ngrady', 'ndgradx', 'ndgrady'])
    # writehsvs('hs-vs_16x16', ['qp', 'ndva', 'ngradx', 'ngrady', 'ndgradx2', 'ndgrady2'])
    # writehsvs('hs-vs_8x8', ['qp', 'ndva', 'ngradx', 'ngrady', 'ndgradx2', 'ndgrady2'])
    # writehsvs('hs-vs_32x16', ['qp', 'ndva', 'ngradx', 'ngrady', 'ndgradx2', 'ndgrady2'])
    # writehsvs('hs-vs_32x8', ['qp', 'ndva', 'ngradx', 'ngrady', 'ndgradx2', 'ndgrady2'])
    # writehsvs('hs-vs_16x8', ['qp', 'ndva', 'ngradx', 'ngrady', 'ndgradx2', 'ndgrady2'])

# 计算均值和方差 sns
if 0:
    # cal_mean_std_sns('s-ns_64x64', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # cal_mean_std_sns('s-ns_32x32', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # cal_mean_std_sns('s-ns_16x16', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # cal_mean_std_sns('s-ns_32x16', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # cal_mean_std_sns('s-ns_32x8', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # cal_mean_std_sns('s-ns_16x8', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    cal_mean_std_sns('s-ns_8x8', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    cal_mean_std_sns('s-ns_32x4', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    cal_mean_std_sns('s-ns_16x4', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    cal_mean_std_sns('s-ns_8x4', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])

# 计算均值和方差 hsvs
if 0:
    # cal_mean_std_hsvs('hs-vs_32x32', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # cal_mean_std_hsvs('hs-vs_16x16', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # cal_mean_std_hsvs('hs-vs_32x16', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # cal_mean_std_hsvs('hs-vs_32x8', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # cal_mean_std_hsvs('hs-vs_16x8', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    cal_mean_std_hsvs('hs-vs_8x8', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])



# Run ParaSearch
if 0:
    ParaSearch = 'libsvmexe/ParaSearch'

# Run ParaSearch 子集
if 0:
    ParaSearch = 'libsvmexe/ParaSearch'
    ParaSearchs = 'libsvmexe/ParaPro'

    # train_sub_set(ParaSearchs, 's-ns_64x64')
    train_sub_set(ParaSearchs, 's-ns_32x32')
    # train_sub_set(ParaSearchs, 's-ns_16x16')
    # # train_sub_set(ParaSearchs, 's-ns_8x8')
    # train_sub_set(ParaSearchs, 's-ns_32x16')
    # train_sub_set(ParaSearchs, 's-ns_32x8')
    # # # train_sub_set(ParaSearchs, 's-ns_32x4')
    # train_sub_set(ParaSearchs, 's-ns_16x8')
    # train_sub_set(ParaSearchs, 's-ns_16x4')
    # train_sub_set(ParaSearchs, 's-ns_8x4')

    # train_sub_set(ParaSearchs, 'hs-vs_32x32')
    # train_sub_set(ParaSearchs, 'hs-vs_16x16')
    # train_sub_set(ParaSearchs, 'hs-vs_8x8')
    # train_sub_set(ParaSearchs, 'hs-vs_32x16')
    # train_sub_set(ParaSearchs, 'hs-vs_32x8')
    # train_sub_set(ParaSearchs, 'hs-vs_16x8')


# Run th
if 0: 
    th = 'libsvmexe/th'
    # select_th(th, 's-ns_64x64')
    # select_th(th, 's-ns_32x32')
    # select_th(th, 's-ns_16x16')
    select_th(th, 's-ns_8x8')
    # select_th(th, 's-ns_32x16')
    # select_th(th, 's-ns_32x8')
    select_th(th, 's-ns_32x4')
    # select_th(th, 's-ns_16x8')
    select_th(th, 's-ns_16x4')
    select_th(th, 's-ns_8x4')
    # select_th(th, 'hs-vs_32x32')
    # select_th(th, 'hs-vs_16x16')
    select_th(th, 'hs-vs_8x8')
    # select_th(th, 'hs-vs_32x16')
    #select_th(th, 'hs-vs_32x8')
    #select_th(th, 'hs-vs_16x8')

# Run th log
if 1:
    readth_log('libsvmdata/')

# Run predict
if 0:
    predict = "libsvmexe/predict"
    for i in range(20):
        run_predict(predict, 'libsvmexe/test.txt', 'libsvmdata/s-ns_16x16/log/model' + str(i) + '.txt', 'libsvmexe/predict' + str(i) + '.txt', 'libsvmexe/log.txt')

# Run log
if 0:
    read_log('libsvmdata/')  

#feature select
if 0:
    #feature_train('s-ns_64x64', ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx'])
    # fe = ['var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx']
    # for f in fe:
    #     feature_train('s-ns_32x16', ['qp', f])
    # feature_train('s-ns_64x64', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # feature_train('s-ns_16x16', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # feature_train('s-ns_32x16', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # feature_train('s-ns_32x8', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # feature_train('s-ns_16x8', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    #feature_train('s-ns_16x4', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    #feature_train('s-ns_8x8', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    #feature_train('s-ns_8x4', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    #feature_train('s-ns_32x4', ['qp', 'var','ngradx', 'ngrady', 'gmx', 'MaxDiffVar', 'MaxDiffgradx', 'MaxDiffgrady'])
    # feature_train('hs-vs_32x32', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # feature_train('hs-vs_32x16', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # feature_train('hs-vs_32x8', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # feature_train('hs-vs_16x16', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    # feature_train('hs-vs_16x8', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])
    feature_train('hs-vs_8x8', ['qp', 'ngradx', 'ngrady', 'ndva', 'ndgradx2', 'ndgrady2'])