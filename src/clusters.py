from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import homogeneity_score
from tqdm import trange, tqdm
import numpy as np


def BestKMeans(X, y, k_min=20, k_max=100):
    best_labels, best = None, 0
    for k in trange(k_min, k_max):
        cluster = KMeans(n_clusters=k, random_state=0, n_init=15)
        labels = cluster.fit_predict(X)
        homogeneity = homogeneity_score(y, labels)
        if homogeneity > best:
            best_labels = labels

    return best_labels


def BestAgglomerative(X, y, k_min=20, k_max=100):
    best_labels, best = None, 0
    for k in trange(k_min, k_max):
        for linkage in ['ward', 'complete', 'average', 'single']:
            for metric in ['euclidean', 'manhattan']:
                cluster = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=linkage,
                    metric=metric
                )
                labels = cluster.fit_predict(X)
                homogeneity = homogeneity_score(y, labels)
                if homogeneity > best:
                    best_labels = labels

    return best_labels


def BestMeanShift(X, y, bandwidth_init, bandwidth_end, step=0.05):
    best_labels, best = None, 0
    for bandwidth in tqdm(np.arange(bandwidth_init, bandwidth_end, step)):
        cluster = MeanShift(bandwidth=bandwidth)
        cluster.fit(X)
        homogeneity = homogeneity_score(y, cluster.labels_)
        if homogeneity > best:
            best_labels = cluster.labels_

    return best, best_labels


def BestDBSCAN(X, y, eps_init, eps_end, step=0.05):
    best_labels, best = None, 0
    for eps in tqdm(np.arange(eps_init, eps_end, step)):
        cluster = DBSCAN(eps=eps)
        cluster.fit(X)
        homogeneity = homogeneity_score(y, cluster.labels_)
        if homogeneity > best:
            best_labels = cluster.labels_

    return best, best_labels


def BestGaussianMixture(X, y, n_comp_init, n_comp_end):
    best_labels, best = None, 0
    for n_components in trange(n_comp_init, n_comp_end):
        cluster = GaussianMixture(n_components=n_components)
        cluster.fit(X)
        label = cluster.predict(X)
        homogeneity = homogeneity_score(y, label)
        if homogeneity > best:
            best_labels = label

    return best, best_labels
