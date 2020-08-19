import numpy as np
import pandas as pd
import math
import minepy as mp

class InformationGain:
    def __init__(self,X,y):
        self.X = np.array(X)
        self.y = y
        self.num_sample = X.shape[0]
        self.num_feature = X.shape[1]
        self.system_entropy = 0
        
        #n_feature = np.shape(X)[1]
        n_y = len(y)

    def cal_total_system_entropy(self):
        for i in set(self.y):
            self.system_entropy += -(self.y.value_counts(1)[i])*math.log(self.y.value_counts(1)[i])
    
    def cal_InformationGain(self):
        self.cal_total_system_entropy()
        InformationGain_list = []
        for i in range(self.num_feature):
            feature_value = self.X[:,i]
            for value in set(feature_value):
                cond_Y = pd.Series([self.y[s] for s in range(len(feature_value)) if feature_value[s] == value])
                cond_H = 0
                print(cond_Y)
                cond_H += -(cond_Y.value_counts(1)[i])*math.log(cond_Y.value_counts(1)[i])
                temp_condi_H = len(cond_Y)/self.num_sample *cond_H
                IG = self.system_entropy - temp_condi_H
                
            InformationGain_list.append(IG)

        return InformationGain_list

if __name__ == "__main__":

    x = np.linspace(0, 1, 1000)
    y = np.sin(10 * np.pi * x) + x
    mine = mp.MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)

    print("Without noise:")
    print("MIC", mine.mic())

    np.random.seed(0)
    y +=np.random.uniform(-1, 1, x.shape[0]) # add some noise
    mine.compute_score(x, y)

    print("With noise:")
    print("MIC", mine.mic())
    
    X = pd.DataFrame([[1,0,0,1],
    [0,1,1,1],
    [0,0,1,0]])
    y = pd.Series([0,0,1])
    print(InformationGain(X,y).cal_InformationGain())