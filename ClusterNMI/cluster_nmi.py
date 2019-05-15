from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import scale, normalize, PolynomialFeatures
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import numpy as np
import math
from ClusterNMI.helpers import get_data, label2result, label2result_DBSCAN


def get_NMI(A, B):
    num_all = len(A)
    set_A = set(A)
    set_B = set(B)

    MI = 0
    # the smallest positive number
    eps = 1.4e-45

    for a in set_A:
        for b in set_B:
            occur_a = np.where(A == a)
            occur_b = np.where(B == b)
            occur_both = np.intersect1d(occur_a, occur_b)
            px = 1.0 * len(occur_a[0])/num_all
            py = 1.0 * len(occur_b[0])/num_all
            pxy = 1.0 * len(occur_both)/num_all
            MI = MI + pxy * math.log2(pxy/(px * py) + eps)

    Hx = 0
    for a in set_A:
        px = 1.0 * len(np.where(A == a)[0])/num_all
        Hx = Hx - px * math.log2(px + eps)

    Hy = 0
    for b in set_B:
        py = 1.0 * len(np.where(B == b)[0]) / num_all
        Hy = Hy - py * math.log2(py + eps)

    NMI = 2.0 * MI/(Hx + Hy)
    return NMI

def preprocess(X):
    X = scale(X)
    X = normalize(X, norm='l2')

    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)
    return X

def KMeans_Clustering(X):
    X = preprocess(X)
    cluster = KMeans(n_clusters=2)
    cluster.fit(X)
    label_pred = cluster.labels_
    return label_pred

def DBSCAN_Clustering(X):
    cluster = DBSCAN(eps=4.4, min_samples=20)
    cluster.fit(X)
    label_pred = cluster.labels_
    return label_pred

def Agglomerative_Clustering(X):
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    cluster = AgglomerativeClustering(connectivity=connectivity, linkage='ward', n_clusters=2)
    cluster.fit(X)
    label_pred = cluster.labels_
    return label_pred

def OPTICS_Clustering(X):
    X = preprocess(X)
    cluster = OPTICS(min_samples=100, xi=.05, min_cluster_size=.05)
    cluster.fit(X)
    label_pred = cluster_optics_dbscan(reachability=cluster.reachability_,
                                   core_distances=cluster.core_distances_,
                                   ordering=cluster.ordering_, eps=2)
    label_pred = cluster.labels_
    return label_pred

def Birch_Clustering(X):
    cluster = Birch(n_clusters=2, threshold=0.5, branching_factor=200)
    cluster.fit(X)
    label_pred = cluster.labels_
    return label_pred

if __name__ == '__main__':
    file_path = "./breast.txt"
    X, label_truth = get_data(file_path)
    label_pred = KMeans_Clustering(X)
    NMI = get_NMI(label_truth, label_pred)
    print(NMI)

    label_res = list(map(label2result, label_pred.tolist()))
    # label_res = list(map(label2result_DBSCAN, label_pred.tolist()))

    with open('./result.txt', 'w') as f:
        for i in label_res:
            f.write(str(i) + '\n')
        f.write("NMI = " + str(NMI))





