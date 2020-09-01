import numpy as np
import numpy as np
import feature_analysis as fa

train_set_path = 'E:\\0-Research\\01-VVC\\Scripts-for-VVC\\vvc9data\\test\\'
df, seq_name= fa.read_csv_data(train_set_path)

model = open('s_ns_square_lib_svm_test.data', 'w')

for i in range(df.shape[0]):
    label = df.loc[i, 'mode']
    if(label == 2000):
        label = 0
    else:
        label = 1
    model.write(str(label) + ' ')
    n = 0
    for m in ['w', 'qp', 'nvar', 'H', 'ngradx', 'ngrady', 'gmx', 'ndvarh', 'ndvarv', 'ndgradxh', 'ndgradyh', 'ndgradxv', 'ndgradyv']:
        model.write(str(n+1) + ':' + str(df.loc[i, m]) + ' ')
        n = n + 1
    model.write('\n')