import numpy as np
from sklearn import svm
from sklearn.externals import joblib

model = open('model.txt', 'w')

clf = joblib.load('square_64_32_16.model')
gamma = 4.45e-08
num_feature = 12

# svm paraments
model.write('svm_type c_svc' + '\n')
model.write('kernel_type rbf' + '\n')
model.write('gamma ' + str(gamma) + '\n')
model.write('nr_class 2' + '\n')
model.write('total_sv ' + str(clf.support_.shape[0]) + '\n')
model.write('rho ' + str(clf.intercept_[0]) + '\n')
model.write('label ' + str(clf.classes_[1]) + ' ' + str(clf.classes_[0]) + '\n')
model.write('probA ' + str(clf.probA_[0]) + '\n')
model.write('probB ' + str(clf.probB_[0]) + '\n')
model.write('nr_sv ' + str(clf.n_support_[0]) + ' ' + str(clf.n_support_[1]) + '\n')
model.write('SV' + '\n')

print(clf.dual_coef_.shape)

# SV
for i in range(clf.support_.shape[0]):
    model.write(str(clf.dual_coef_[0][i]) + ' ')
    for m in range(num_feature):
        model.write(str(m+1) + ':' + str(clf.support_vectors_[i][m]) + ' ')
    model.write('\n')


model.close()