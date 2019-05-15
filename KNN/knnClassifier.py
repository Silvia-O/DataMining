from initData import get_data, get_data_optimized
from helpers import dist, plot
import numpy as np


def knn_classify(data_file, K=5):
    # f_train, f_test, l_train, l_test = get_data(data_file)
    f_train, f_test, l_train, l_test = get_data_optimized(data_file)

    l_predict = []
    for t in range(f_test.shape[0]):
        N = []
        l = 0
        for d in range(f_train.shape[0]):
            if len(N) <= K:
                N.append(d)
            else:
                for u in N:
                    if dist(f_test[t], f_train[d]) < dist(f_test[t], f_train[u]):
                        N.remove(u)
                        N.append(d)
                        break
        for n in N:
            if l_train[n] == 'A':
                l += 1
            else:
                l -= 1
        if l >= 0:
            l_predict.append('A')
        else:
            l_predict.append('B')
    acc = (float(sum(l_predict == l_test))/len(l_predict))
    plot(f_train, l_train, f_test, l_test, l_predict, acc)


if __name__ == "__main__":
    data_file = './dataset_data_mining_course.csv'
    knn_classify(data_file)
