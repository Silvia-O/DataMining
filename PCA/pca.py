import numpy as np
import matplotlib.pyplot as plt


def pca(data, N):
    mean_data = np.mean(data, axis=0)
    mean_removed = data - mean_data

    # cov matrix
    covMat = np.dot(np.transpose(mean_removed), mean_removed)
    eig_val, eig_vec = np.linalg.eig(covMat)

    eig_val_idx = np.argsort(eig_val)
    # choose N eig values
    eig_val_idx = eig_val_idx[:-(N + 1):-1]
    red_eig_vec = eig_vec[:, eig_val_idx]

    # get lower dim data
    low_dim_data = np.dot(mean_removed, red_eig_vec)
    return low_dim_data


def show(data, N):
    plt.figure()
    if N == 2:
        plt.scatter(data[:, 0].flatten(), data[:, 1].flatten(), marker='^', s=90)
    elif N == 1:
        plt.scatter(range(0, np.size(data,0)), data, marker='o',s=90,c='red')
    plt.title(str(N) + '-dim data')
    plt.show()


data = np.loadtxt('dataset2_data_mining_course.csv', delimiter=',', usecols=(0,1,2))
# remove label

data1 = pca(data, 2)
show(data1, 2)
data2 = pca(data, 1)
show(data2, 1)
