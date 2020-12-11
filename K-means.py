import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    set ground true cluster parameters, here we set 3 different clusters
    n_samples:  a 1x3 matrix containing sample numbers in each clusters
    n_features: is the dimension of sample space
    centers: the mean of each clusters
    cluster_std: the stand error of each clusters
    X: shape like (n_samples.sum(), n_features) 
    y: shape like(, n_samples.sum())
    """
    X, y = make_blobs(n_samples=[73, 70, 12],
                      n_features=2,
                      centers=[[3, 4], [4, 5.2], [4.7, 3.5]],
                      cluster_std=0.6)
    # initialize figure
    plt.figure(figsize=(10, 10))

    # plot ground true
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Ground True")

    # make iteration for 3-time different clustering
    seq = [222, 223, 224]

    for s in seq:
        # using k-means method and random initialization to clustering
        y_pred = KMeans(n_clusters=3).fit_predict(X)

        # plot the clustering result
        plt.subplot(s)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.title("K-means cluster")

        # compute the sum of square error

        squareError = 0
        for c in range(3):
            loc = np.where(y_pred == c)
            cluster = X[loc]
            squareError += np.square(cluster - cluster.mean())

    squareError_true = 0
    for c in range(3):
        loc = np.where(y == c)
        cluster = X[loc]
        squareError_true += np.square(cluster - cluster.mean())

    plt.show()
