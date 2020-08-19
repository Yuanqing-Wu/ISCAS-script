import glob
import os
import pandas as pd

read_path = 'E:\\0-Research\\01-VVC\\result\\ALL\\'      # 要读取的文件夹的地址

#fdata_analysis = open('fdata_analysis.csv', 'w')

def dic_write(df, w, h):
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



read_csv = glob.glob(os.path.join(read_path,'*.csv')) # 读取文件夹中所有后缀为xlsx的文件地址
df = None
for i, path in enumerate(read_csv):     # 循环读取所有后缀为xlsx的文件

    #before, file_name = path.split(read_path)
    #file_name, csv = file_name.split('.')
    #df = pd.read_csv(path)
    month = pd.read_csv(path)
    if df is None:          # 第一次df为空，需要赋值为DataFrame
        df = month
    else:
        df = pd.concat([df,month], ignore_index = True)  # 之后读取的每个文件都与前一个文件合并


def write_split_result(file_name, seq_name, df):
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