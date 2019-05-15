import numpy as np
import matplotlib.pyplot as plt


def dist(m, n):
    return np.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2)


def plot(f_train, l_train, f_test, l_test, l_predict, acc, test_size=0.9):
    c1_train, c2_train = split(f_train, l_train)
    c1_test, c2_test = split(f_test, l_test)
    c1_pred, c2_pred = split(f_test, l_predict)
    fig = plt.figure(22)
    p1 = fig.add_subplot(221)
    p2 = fig.add_subplot(223)
    p3 = fig.add_subplot(224)
    fig.set_size_inches(30, 20)
    plt.xlabel('X')
    plt.ylabel('Y')

    p1.set_title("Before Cassifying, Test size = %f" % test_size)
    p1.scatter(c1_train[:, 0], c1_train[:, 1], c='r', marker='x')
    p1.scatter(c2_train[:, 0], c2_train[:, 1], c='y', marker='x')
    p1.scatter(f_test[:, 0], f_test[:, 1], c='c', marker='o')

    p2.set_title("After Classifying, Acc = %f" % acc)
    p2.scatter(c1_train[:, 0], c1_train[:, 1], c='r', marker='x')
    p2.scatter(c2_train[:, 0], c2_train[:, 1], c='y', marker='x')
    p2.scatter(c1_pred[:, 0], c1_pred[:, 1], c='r', marker='o')
    p2.scatter(c2_pred[:, 0], c2_pred[:, 1], c='y', marker='o')

    p3.set_title("Ground Truth")
    p3.scatter(c1_train[:, 0], c1_train[:, 1], c='r', marker='x')
    p3.scatter(c2_train[:, 0], c2_train[:, 1], c='y', marker='x')
    p3.scatter(c1_test[:, 0], c1_test[:, 1], c='r', marker='o')
    p3.scatter(c2_test[:, 0], c2_test[:, 1], c='y', marker='o')

    # plt.savefig('knn_result.png')
    plt.savefig('knn_optimized_result.png', dpi=100)
    plt.show()


def split(f, l):
    num = f.shape[0]
    c1 = np.zeros((num, 2))
    c2 = np.zeros((num, 2))
    m = 0
    n = 0
    for i in range(num):
        if l[i] == 'A':
            c1[m] = f[i]
            m += 1
        else:
            c2[n] = f[i]
            n += 1
    return c1, c2