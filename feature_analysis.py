import glob
import os
import numpy as np
import pandas as pd
import minepy as mp
import matplotlib.pyplot as plt

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

def cal_mic(x, y):

    # calculate the maximal information coefficient 

    x = np.array(x)
    y = np.array(y)
    mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)

    return mine.mic()

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

def save_block_set(df, w, h, file_name, data_size = 0):
    df = df[df['w'] == w]
    df = df[df['h'] == h]

    if data_size == 0:
        df = balance_set(df[df['mode'] == 2000], df[df['mode'] != 2000])
    else:
        df = balance_set(df[df['mode'] == 2000], df[df['mode'] != 2000], data_size)
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



# feature = ['w', 'h', 'mode', 'qp', 'qt_d', 'mt_d', 'var', 'H', 'gradx', 'grady', 'maxgrad', 'dvarh', 'dvarv', 'dHh', 'dHv', 'dgradxh', 'dgradxv', 'dgradyh', 'dgradyv']


if __name__ == "__main__":

    read_path = 'E:\\0-Research\\01-VVC\\result\\train\\'      # the path of csv file
    #read_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\train\\'
    df, seq_name = read_csv_data(read_path)
    #df = df.loc[:, ['mode', 'w', 'qp', 'nvar', 'H', 'ngradx', 'ngrady', 'gmx', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']]

    save_block_set(df, 32, 16,'s-ns_train', 1000)
    # save_block_set(df, 32, 8,'s-ns_rectangle_train', 1000)
    # save_block_set(df, 32, 4,'s-ns_rectangle_train', 1000)

    # save_block_set(df, 16, 8,'s-ns_rectangle_train', 1000)
    # save_block_set(df, 16, 4,'s-ns_rectangle_train', 1000)

    # save_block_set(df, 8, 8,'s-ns_rectangle_train', 1000)
    # save_block_set(df, 8, 4,'s-ns_rectangle_train', 1000)
    # write_split_result('10bit_all.csv', 'cu', df)
    #block_static(df, 32, 16, 27, 'nvar', 2000, [-5000, 50000])