import numpy as np
import pandas as pd

def get_data(data_file, test_size=0.9):
    data = pd.read_csv(data_file, delimiter=" ")
    data = np.array(data)
    feature = data[:, :-1]
    label = data[:, -1]
    return feature, label


def label2result(x):
    if(x == 0):
        return 2
    elif(x == 1):
        return 4

def label2result_DBSCAN(x):
    if (x == -1):
        return 2
    elif (x == 0):
        return 4

