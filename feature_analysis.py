import glob
import os
import numpy as np
import pandas as pd
import minepy as mp
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split

def dic_write(df, w, h):

    # write split probility of everu cu size to dictionary

    df = df[df['w'] == w]
    df = df[df['h'] == h]
    num_split = df["mode"].value_counts(1) 
    if 2000 in num_split:
        split_ratio_no = num_split[2000]
    else:
        split_ratio_no = 0

    if 1 in num_split:
        split_ratio_qt = num_split[1] 
    else:
        split_ratio_qt = 0

    if 2 in num_split:
        split_ratio_bh = num_split[2] 
    else:
        split_ratio_bh = 0

    if 3 in num_split:    
        split_ratio_bv = num_split[3]
    else:  
        split_ratio_bv = 0

    if 4 in num_split:    
        split_ratio_th = num_split[4]
    else:  
        split_ratio_th = 0
    
    if 5 in num_split:    
        split_ratio_tv = num_split[5]
    else:  
        split_ratio_tv = 0
  
    return [split_ratio_no, split_ratio_qt, split_ratio_bh, split_ratio_bv, split_ratio_th, split_ratio_tv]


def write_split_result(file_name, seq_name, df):

    # write split result to csv file

    data = {}

    data[seq_name + '_64x64'] = dic_write(df, 64, 64)
    data[seq_name + '_32x32'] = dic_write(df, 32, 32)
    data[seq_name + '_32x16'] = dic_write(df, 32, 16)
    data[seq_name + '_32x8' ] = dic_write(df, 32,  8)
    data[seq_name + '_32x4' ] = dic_write(df, 32,  4)
    data[seq_name + '_16x32'] = dic_write(df, 16, 32)
    data[seq_name + '_8x32' ] = dic_write(df,  8, 32)
    data[seq_name + '_4x32' ] = dic_write(df,  4, 32)
    data[seq_name + '_16x16'] = dic_write(df, 16, 16)
    data[seq_name + '_16x8' ] = dic_write(df, 16,  8)
    data[seq_name + '_16x4' ] = dic_write(df, 16,  4)
    data[seq_name + '_8x16' ] = dic_write(df,  8, 16)
    data[seq_name + '_4x16' ] = dic_write(df,  4, 16)
    data[seq_name + '_8x8'  ] = dic_write(df,  8,  8)
    data[seq_name + '_8x4'  ] = dic_write(df,  8,  4)
    data[seq_name + '_4x8'  ] = dic_write(df,  4,  8)

    dd = pd.DataFrame(data)
    # dd['avr'] = dd.mean(1)
    # print(dd)
    pd.DataFrame(dd.values.T, index=dd.columns, columns=dd.index).to_csv(file_name)



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

def mic(x, y):

    # calculate the maximal information coefficient 

    x = np.array(x)
    y = np.array(y)
    mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)

    return mine.mic()

def cal_mic(df, feature):

    # calculate the maximal information coefficient 
    pd_mic = pd.DataFrame(index=feature, columns=feature)
    for f1 in feature:
        for f2 in feature:
            x = np.array(df.loc[:, f1])
            y = np.array(df.loc[:, f2])
            pd_mic.loc[f1, f2] = mic(x, y)
    pd_mic['avr'] = pd_mic.mean(1)
    #print(pd_mic)
    pd_mic.to_csv('mic.csv')

def cal_mi(df, feature):

    X = df.loc[:, feature]
    y = np.array(df.loc[:, 'mode'])
    print(mutual_info_classif(X,y))



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

def save_block_set_hsvs(df, w, h, file_name, data_size = 0, balance_set = True):
    df = df[df['w'] == w]
    df = df[df['h'] == h]

    for i in df.index:
        if df.loc[i, 'mode'] == 2 or df.loc[i, 'mode'] == 4:
            df.loc[i, 'mode'] = 100  # HS
        if df.loc[i, 'mode'] == 3 or df.loc[i, 'mode'] == 5:
            df.loc[i, 'mode'] = 200  # VS
    
    if balance_set:
        if data_size == 0:
            df = balance_set(df[df['mode'] == 100], df[df['mode'] == 200])
        else:
            df = balance_set(df[df['mode'] == 100], df[df['mode'] == 200], data_size)
    file_name = file_name + '_' + str(w) + 'x' +str(h) + '_' +str(data_size) + '.csv'
    df.to_csv(file_name)


def block_static(df, w, h, qp, feature, label, xlim, normalize = False):

    df = df[df['w'] == w]
    df = df[df['h'] == h]
    df = df[df['qp'] == qp]

    df = balance_set(df[df['mode'] == label], df[df['mode'] != label])

    if normalize == True:
        df[feature] = np.divide(df[feature], w*h)

    A1 = df[df['mode'] == label]
    A2 = df[df['mode'] != label]

    B1 = A1.loc[:, [feature]]
    B2 = A2.loc[:, [feature]]

    plt.figure(num=feature)
    ax = B1.plot(kind='kde')
    B2.plot(kind='kde', ax=ax, xlim=xlim)
    plt.savefig(fname = feature + '.png')

def size_reuse_size(size, *args):
    n = len(args)
    strr = ''
    for i in range(n):
        if i != n-1:
            strr= strr + args[i] + size 
        else:
            strr = strr + args[i]
    print(strr)
    return strr

def size_reuse(*args):
    strr = []
    strr.append(size_reuse_size('64x64', *args))
    strr.append(size_reuse_size('32x32', *args))
    strr.append(size_reuse_size('16x16', *args))
    strr.append(size_reuse_size('8x8', *args))
    strr.append(size_reuse_size('32x16', *args))
    strr.append(size_reuse_size('32x8', *args))
    strr.append(size_reuse_size('32x4', *args))
    strr.append(size_reuse_size('16x8', *args))
    strr.append(size_reuse_size('16x4', *args))
    strr.append(size_reuse_size('8x4', *args))
    return strr

def stand_sca(data):
    """
    标准差标准化
    :param data:传入的数据
    :return:标准化之后的数据
    """
    new_data=(data-data.mean())/data.std()
    return new_data
    



# feature = ['w', 'h', 'mode', 'qp', 'qt_d', 'mt_d', 'var', 'H', 'gradx', 'grady', 'maxgrad', 'dvarh', 'dvarv', 'dHh', 'dHv', 'dgradxh', 'dgradxv', 'dgradyh', 'dgradyv']


if __name__ == "__main__":
    # save_block_set_hsvs(df, 32, 32,'hs-vs', balance_set = False)
    # save_block_set_hsvs(df, 16, 16,'hs-vs', balance_set = False)
    # save_block_set_hsvs(df, 8, 8,'hs-vs', balance_set = False)
    # save_block_set_hsvs(df, 32, 16,'hs-vs', balance_set = False)
    # save_block_set_hsvs(df, 32, 8,'hs-vs', balance_set = False)
    # save_block_set_hsvs(df, 16, 8,'hs-vs', balance_set = False)


    
    # 保存划分结果
    if 0:
        read_path = 'E:\\0-Research\\00-ISCAS\\DataSet\\Test\\'      # the path of csv file
        df, seq_name = read_csv_data(read_path)  
        df = df[df['chType'] == 0]
        write_split_result('10bit_all.csv', 'cu', df)

    # 保存块data
    if 1:
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
    
    # 保存块data (1帧)
    if 0:
        read_path = 'E:\\0-Research\\00-ISCAS\\DataSet\\SingleFrame\\'      # the path of csv file
        df, seq_name = read_csv_data(read_path)
    
        df = df[df['chType'] == 0]
        save_block_set_sns(df, 64, 64,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 32, 32,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 16, 16,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 8, 8,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 32, 16,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 32, 8,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 32, 4,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 16, 8,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 16, 4,'test-s-ns', isbalance_set = False)
        save_block_set_sns(df, 8, 4,'test-s-ns', isbalance_set = False)

    # 计算mic
    if 0:
        train_set_path = 'E:\\0-Research\\00-ISCAS\\PyScript\\csv_data\\s-ns_32x32_0.csv'    
        df = pd.read_csv(train_set_path)

        y = df.loc[:, 'mode']
        y[y != 2000] = 1
        y[y == 2000] = 0
        
        feature = ['qp', 'var', 'ndvarh','ndvarv','ndva','MaxDiffVar','InconsVarH','InconsVarV','ngradx','ngrady','ndgradxh','ndgradxv','ndgradyh','ndgradyv','ndgradx','ndgrady','gmx']
        for f in feature:
            print(f + ": ", mic(y, df.loc[:, f]))            
    
    # 数据预处理：标准化
    if 0:
        train_set_path = 'E:\\0-Research\\00-ISCAS\\PyScript\\csv_data\\s-ns_64x64_0.csv'    
        df = pd.read_csv(train_set_path)
        X = df.loc[:, ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx']]
        print(X)
        X = stand_sca(X)
        print(X)
    
    # 标准化训练
    if 0:
        train_set_path = 'E:\\0-Research\\00-ISCAS\\PyScript\\csv_data\\s-ns_32x32_0.csv'    
        df = pd.read_csv(train_set_path)
        X = df.loc[:, ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx']]
        X2 = stand_sca(X)

        y = df.loc[:, 'mode']
        y[y != 2000] = 1
        y[y == 2000] = 0

        X_train, _, y_train, _ = train_test_split(X, y, train_size = 500, random_state = 1, stratify = y)

        svc = svm.SVC(kernel='rbf', C = 100, probability = True)
        svc.fit(X_train, y_train)
        print('no: ', svc.score(X, y))
    
    # Parament Search
    if 0:
        train_set_path = 'E:\\0-Research\\00-ISCAS\\PyScript\\csv_data\\s-ns_32x32_0.csv'    
        df = pd.read_csv(train_set_path)
        X = df.loc[:, ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx']]
        X2 = stand_sca(X)

        y = df.loc[:, 'mode']
        y[y != 2000] = 1
        y[y == 2000] = 0

        X_train, _, y_train, _ = train_test_split(X, y, train_size = 500, random_state = 1, stratify = y)

        C = [10, 100, 1000, 10000, 100000]
        gamma = [0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.000000000001]
        for c in C:
            for g in gamma:
                svc = svm.SVC(kernel='rbf', C = c, gamma = g, probability = True)
                svc.fit(X_train, y_train)
                print(c, g, svc.score(X, y))

    # 非标准化训练
    if 0:
        train_set_path = 'E:\\0-Research\\00-ISCAS\\PyScript\\csv_data\\s-ns_64x64_0.csv' 
        df = pd.read_csv(train_set_path)
        X = df.loc[:, ['qp', 'var', 'ngradx', 'ngrady', 'MaxDiffVar', 'InconsVarH', 'InconsVarV', 'gmx']]
        y = df.loc[:, 'mode']
        y[y != 2000] = 1
        y[y == 2000] = 0

        X_train, _, y_train, _ = train_test_split(X, y, train_size = 500, random_state = 1, stratify = y)

        svc = svm.SVC(kernel='rbf', C = 100000, probability = True)
        svc.fit(X_train, y_train)
        print('no: ', svc.score(X, y))
        #print(y_pre)

