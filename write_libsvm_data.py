import numpy as np
import pandas as pd
from feature_analysis import *

train_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\train\\'
test_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\test\\'



def write_data(data_file, df, feature):

    for i in range(df.shape[0]):
        label = df.loc[i, 'mode']
        if(label == 2000):
            label = 0
        else:
            label = 1
        data_file.write(str(label) + ' ')
        n = 0
        for m in feature:
            data_file.write(str(n+1) + ':' + str(df.loc[i, m]) + ' ')
            n = n + 1
        data_file.write('\n')

def writeall(train_set_path, test_set_path):

    path_64x64 = train_set_path + 's-ns_train_64x64_50.csv'
    path_32x32 = train_set_path + 's-ns_train_32x32_50.csv'
    path_16x16 = train_set_path + 's-ns_train_16x16_50.csv'
    path_8x8 = train_set_path + 's-ns_train_8x8_30.csv'
    path_32x16 = train_set_path + 's-ns_train_32x16_30.csv'
    path_32x8 = train_set_path + 's-ns_train_32x8_30.csv'
    path_32x4 = train_set_path + 's-ns_train_32x4_30.csv'
    path_16x8 = train_set_path + 's-ns_train_16x8_30.csv'
    path_16x4 = train_set_path + 's-ns_train_16x4_20.csv'
    path_8x4 = train_set_path + 's-ns_train_8x4_20.csv'

    df_64x64 = pd.read_csv(path_64x64)
    df_32x32 = pd.read_csv(path_32x32)
    df_16x16 = pd.read_csv(path_16x16)
    df_8x8 = pd.read_csv(path_8x8)
    df_32x16 = pd.read_csv(path_32x16)
    df_32x8 = pd.read_csv(path_32x8)
    df_32x4 = pd.read_csv(path_32x4)
    df_16x8 = pd.read_csv(path_16x8)
    df_16x4 = pd.read_csv(path_16x4)
    df_8x4 = pd.read_csv(path_8x4)

    test_path_64x64 = test_set_path + 's-ns_test_64x64_0.csv'
    test_path_32x32 = test_set_path + 's-ns_test_32x32_0.csv'
    test_path_16x16 = test_set_path + 's-ns_test_16x16_0.csv'
    test_path_8x8 = test_set_path + 's-ns_test_8x8_0.csv'
    test_path_32x16 = test_set_path + 's-ns_test_32x16_0.csv'
    test_path_32x8 = test_set_path + 's-ns_test_32x8_0.csv'
    test_path_32x4 = test_set_path + 's-ns_test_32x4_0.csv'
    test_path_16x8 = test_set_path + 's-ns_test_16x8_0.csv'
    test_path_16x4 = test_set_path + 's-ns_test_16x4_0.csv'
    test_path_8x4 = test_set_path + 's-ns_test_8x4_0.csv'

    test_df_64x64 = pd.read_csv(test_path_64x64)
    test_df_32x32 = pd.read_csv(test_path_32x32)
    test_df_16x16 = pd.read_csv(test_path_16x16)
    test_df_8x8 = pd.read_csv(test_path_8x8)
    test_df_32x16 = pd.read_csv(test_path_32x16)
    test_df_32x8 = pd.read_csv(test_path_32x8)
    test_df_32x4 = pd.read_csv(test_path_32x4)
    test_df_16x8 = pd.read_csv(test_path_16x8)
    test_df_16x4 = pd.read_csv(test_path_16x4)
    test_df_8x4 = pd.read_csv(test_path_8x4)

    data_64x64_file = open('s_ns_64x64_train.data', 'w')
    data_32x32_file = open('s_ns_32x32_train.data', 'w')
    data_16x16_file = open('s_ns_16x16_train.data', 'w')
    data_8x8_file = open('s_ns_8x8_train.data', 'w')
    data_32x16_file = open('s_ns_32x16_train.data', 'w')
    data_32x8_file = open('s_ns_32x8_train.data', 'w')
    data_32x4_file = open('s_ns_32x4_train.data', 'w')
    data_16x8_file = open('s_ns_16x8_train.data', 'w')
    data_16x4_file = open('s_ns_16x4_train.data', 'w')
    data_8x4_file = open('s_ns_8x4_train.data', 'w')

    test_data_64x64_file = open('s_ns_64x64_test.data', 'w')
    test_data_32x32_file = open('s_ns_32x32_test.data', 'w')
    test_data_16x16_file = open('s_ns_16x16_test.data', 'w')
    test_data_8x8_file = open('s_ns_8x8_test.data', 'w')
    test_data_32x16_file = open('s_ns_32x16_test.data', 'w')
    test_data_32x8_file = open('s_ns_32x8_test.data', 'w')
    test_data_32x4_file = open('s_ns_32x4_test.data', 'w')
    test_data_16x8_file = open('s_ns_16x8_test.data', 'w')
    test_data_16x4_file = open('s_ns_16x4_test.data', 'w')
    test_data_8x4_file = open('s_ns_8x4_test.data', 'w')

    write_data(data_64x64_file, df_64x64, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(data_32x32_file, df_32x32, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(data_16x16_file, df_16x16, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(data_8x8_file, df_8x8, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(data_32x16_file, df_32x16, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(data_32x8_file, df_32x8, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(data_32x4_file, df_32x4, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxv', 'ndgradyv'])
    write_data(data_16x8_file, df_16x8, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(data_16x4_file, df_16x4, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxv', 'ndgradyv'])
    write_data(data_8x4_file, df_8x4, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxv', 'ndgradyv'])

    write_data(test_data_64x64_file, test_df_64x64, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_32x32_file, test_df_32x32, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_16x16_file, test_df_16x16, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_8x8_file, test_df_8x8, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_32x16_file, test_df_32x16, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_32x8_file, test_df_32x8, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_32x4_file, test_df_32x4, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_16x8_file, test_df_16x8, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_16x4_file, test_df_16x4, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxv', 'ndgradyv'])
    write_data(test_data_8x4_file, test_df_8x4, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxv', 'ndgradyv'])

path_32x4 = train_set_path + 's-ns_train_32x4_40.csv'
df_32x4 = pd.read_csv(path_32x4)
data_32x4_file = open('s_ns_32x4_train.data', 'w')
write_data(data_32x4_file, df_32x4, ['qp', 'ngradx', 'ngrady', 'gmx', 'ndgradxv', 'ndgradyv'])
