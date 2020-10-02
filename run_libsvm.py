import os
import subprocess

train = "libsvmexe\\train"
predict = "libsvmexe\\predict"
ParaSearch = 'libsvmexe\\ParaSearch'
th = 'libsvmexe\\th'


def run_one(exe, size):

    train_file = 'libsvmdata\\s_ns_' + size + '_train.data'
    test_file = 'libsvmdata\\s_ns_' + size + '_test.data'
    model_file = 'libsvmmodel\\s_ns_' + size + '.model'
    predict_file = 'libsvmdata\\s_ns_' +  size + '_predict.data'

    before, exename = exe.split('\\')

    cmd = ''
    if exe == ParaSearch:
        cmd = exe + ' -i ' + train_file + ' -e ' + test_file + ' -m ' + model_file + ' -p ' + predict_file + ' > libsvmmodel\\s_ns_train' + size + exename + '.log'
    if exe == th:
        cmd = exe + ' -e ' + test_file + ' -m ' + model_file + ' -p ' + predict_file + ' > libsvmmodel\\s_ns_train' + size + exename + '.log'
    subprocess.Popen(cmd, shell=True)

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

run_all(th)





