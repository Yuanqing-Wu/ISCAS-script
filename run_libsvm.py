import os
import subprocess

train_exe = "libsvmexe\\train"
predict_exe = "libsvmexe\\predict"
my_exe = 'libsvmexe\\my'


def run_one(exe, size):

    train_file = 'libsvmdata\\s_ns_' + size + '_train.data'
    test_file = 'libsvmdata\\s_ns_' + size + '_test.data'
    model_file = 'libsvmmodel\\s_ns_' + size + '.model'
    predict_file = 'libsvmdata\\s_ns_' +  size + '_predict.data'

    cmd = my_exe + ' -i ' + train_file + ' -e ' + test_file + ' -m ' + model_file + ' -p ' + predict_file + ' > libsvmmodel\\s_ns_train' + size + '.log'
    proc = subprocess.Popen(cmd, shell=True)

def run_all(exe):

    run_one(exe, '64x64')
    run_one(exe, '32x32')
    run_one(exe, '16x16')
    run_one(exe, '8x8')
    run_one(exe, '32x16')
    run_one(exe, '32x8')
    run_one(exe, '32x4')
    run_one(exe, '16x8')
    run_one(exe, '16x4')
    run_one(exe, '8x4')

run_all(my_exe)





