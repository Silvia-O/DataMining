import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_splitminsample
from helpers import dist
import random


def get_data(data_file, test_size=0.9):
    data = pd.read_csv(data_file)
    data = np.array(data)
    feature = data[:, :-1]
    label = data[:, -1]
    f_train, f_test, l_train, l_test = train_test_split(feature, label, test_size=test_size)

    return f_train, f_test, l_train, l_test


def get_data_optimized(data_file, test_size=0.9):
    data = pd.read_csv(data_file)
    data = np.array(data)
    train_data = []
    seed_point = random.randint(0, data.shape[0]-1)
    train_data.append(data[seed_point])
    test_data = np.delete(data, seed_point, axis=0)

    while len(train_data) <= (int((1 - test_size) * data.shape[0])):
        dist_buf = []
        for i in range(test_data.shape[0]):
            min_dist = float('inf')
            for j in range(len(train_data)):
                d = dist(train_data[j], test_data[i])
                if d < min_dist:
                    min_dist = d
            dist_buf.append(min_dist)
        max_point = dist_buf.index(max(dist_buf))
        train_data.append(test_data[max_point])
        test_data = np.delete(test_data, max_point, axis=0)

    train_data = np.array(train_data)
    f_train = train_data[:, :-1]
    l_train = train_data[:, -1]
    f_test = test_data[:, :-1]
    l_test = test_data[:, -1]
    return f_train, f_test, l_train, l_test