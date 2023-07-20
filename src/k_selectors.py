from math import log, sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from functools import lru_cache
from tqdm import tqdm


class DistanceManager:
    def __init__(self, data, metric) -> None:
        self.metric = metric
        self.data = data
        self.indexes = np.array(range(len(data)))
        self.memory = {}

    def intra_sum(self, labels):
        DW = 0
        for n_cluster in set(labels):
            DW += self._intra_sum(frozenset(self.indexes[labels == n_cluster]))
        return DW

    @lru_cache
    def _intra_sum(self, indexes):
        indexes = list(indexes)
        return sum([
            self.__d(x, y)
            for i, x in enumerate(indexes[:-1])
            for y in indexes[i+1:]
        ])

    @lru_cache
    def __d(self, i, j):
        try:
            return self.memory[(i, j)]
        except KeyError:
            self.memory[(j, i)] = pairwise_distances(
                self.data[i].reshape(1, -1),
                self.data[j].reshape(1, -1),
                metric=self.metric
            )[0][0]
            self.memory[(i, j)] = self.memory[(j, i)]
            return self.memory[(i, j)]

    def __len__(self):
        return len(self.data)


def k_selector_pipeline(cluster_func, selector, k_max=20, data=None):
    K = list(range(2, k_max))
    labels = []
    for k in tqdm(K):
        labels.append(cluster_func(k, data=data))

    optimal_ks = []
    for s in selector:
        print(f'Analyse by {s.__class__.__name__}')
        optimal_ks.append(s.fit_predict(K, labels))

    return optimal_ks


class Hartigan:
    def __init__(self, dist: DistanceManager, tol=10) -> None:
        self.dist = dist
        self.tol = tol

    def fit_predict(self, ks, labels):
        self.K = np.array(ks)
        self.SW = np.array([
            self.dist.intra_sum(label)
            for label in tqdm(labels)
        ])

        self.H = (self.SW[0:-2] / self.SW[1:-1] - 1) * \
            (len(self.dist) - self.K[0:-2] - 1)

        for i, h in enumerate(self.H):
            if h <= self.tol:
                return self.K[i]

        return self.K[-1]


class Silhouette:
    def __init__(self, data):
        self.data = data

    def fit_predict(self, ks, labels):
        silhouette, K = [], []
        for k, label in tqdm(zip(ks, labels)):
            if self.k <= 1:
                return 0
            silhouette.append(silhouette_score(self.data, label))
            K.append(k)

        return K[np.argmax(silhouette)]


class GapStatistics:
    def __init__(self, dist: DistanceManager, b_size=20):
        self.dist = dist
        self.b_size = b_size

    def fit_predict(self, ks, labels):
        gaps, gaps_s = [], []
        for k, label in tqdm(zip(ks, labels)):
            Wk = self.dist.intra_sum(label)
            w_gap = np.zeros(self.b_size)

            for i in range(self.b_size):
                w_gap[i] = self.dist.intra_sum(
                    np.random.randint(0, k, size=len(self.dist))
                )

            log_w_gap = np.log(w_gap)
            gap_value = np.mean(log_w_gap) - np.log(Wk)
            sdk = np.sqrt(
                np.mean((log_w_gap - np.mean(log_w_gap)) ** 2.0)
            )
            sk = np.sqrt(1.0 + 1.0 / self.b_size) * sdk

            gaps.append(gap_value)
            gaps_s.append(gap_value * sk)

        for optime in ks[:-1]:
            if gaps[optime] >= gaps_s[optime + 1]:
                break

        return optime


# class KSelector:
#     def __init__(self):
#         self.results = {}

#     def k(self, labels):
#         return len(set(list(labels)))

#     def save(self, labels, result):
#         self.results[self.k(labels)] = result
#         return result

#     def map(self):
#         return list(self.results.keys()), list(self.results.values())

#     def plot(self):
#         X, Y = self.map()
#         fig, ax = plt.subplots()
#         ax.plot(X, Y, marker='o')
#         ax.set_title(self.__class__.__name__)
#         ax.set_xlabel("CN")
#         ax.set_ylabel("Heuristic Value")


# class GapStatitics(ElbowMethod):
#     def __init__(self, distances, b_size=20):
#         super().__init__(distances)
#         self.b_size = b_size

#     def map(self):


#     def plot(self):


#         plt.axvline(x=optime + 1, color='b', label='K Optimo')

#         # Plotear la estadística de gap
#         plt.plot(ks, gaps, '-o', label='Gap')
#         plt.plot(ks, gaps_s, '-o', label='Gap * s')
#         plt.xlabel('Cantidad de Grupos')
#         plt.ylabel('Gap Statistic')
#         plt.title('Método del Gap-Statistic para KMeans')
#         plt.legend()
#         plt.show()


class Density:
    def __init__(self, classification, tol=0.7):
        self.classification = classification
        self.tol = tol

    def fit(self, labels):
        max_label = np.max(labels)
        X = list(range(max_label))

        self.ones, self.zeros = np.zeros(len(X)), np.zeros(len(X))
        for i, x in enumerate(X):
            self.ones[i] = np.sum(self.classification[labels == x] == 1)
            self.zeros[i] = np.sum(self.classification[labels == x] == 0)

        totals = self.ones + self.zeros
        maximals = np.max([self.ones, self.zeros], axis=1)
        s = np.sum((maximals/totals) >= self.tol)

        return s / max_label

    def plot_last(self, labels):
        data = pd.DataFrame({
            'labels': labels,
            'count': [1] * len(labels),
            'classification': self.classification
        })

        sns.set(style='whitegrid')
        sns.barplot(
            x='labels',
            y='count',
            data=data,
            hue='classification',
            estimator=np.sum,
        )
        plt.title('Homogeneidad de los clusters')
        plt.show()
