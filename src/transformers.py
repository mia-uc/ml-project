from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statistics as stats
import prince
from sklearn.preprocessing import OneHotEncoder


class Transformer(ABC):
    def __init__(self, name) -> None:
        self.name = name

    def fit(self, X):
        try:
            return pd.read_csv(self.name)
        except:
            return None

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, *args):
        r = self.fit(X)
        if not r is None:
            return r
        return self.transform(X)


class CountTransformer(Transformer):
    def __init__(self, max_index, save=True) -> None:
        super().__init__('count.csv')
        self.max_index = max_index
        self.count_columns = [f'count_{i+1}' for i in range(max_index)]
        self.save = save

    def transform(self, sequences):
        count_host = np.zeros((len(sequences), self.max_index))
        for i in trange(self.max_index):
            for j, seq in enumerate(sequences):
                count_host[j, i] = sum([i + 1 == x for x in seq])

        _df = pd.DataFrame(count_host, columns=self.count_columns)
        if self.save:
            _df.to_csv(self.name, index=False)

        return _df


class FirstTransformer(Transformer):
    def __init__(self, max_index, save=True) -> None:
        super().__init__('first.csv')
        self.max_index = max_index
        self.columns = [f'first_{i+1}' for i in range(max_index)]
        self.save = save

    def transform(self, sequences):
        first_host = np.zeros((len(sequences), self.max_index))
        for i in trange(self.max_index):
            for j, seq in enumerate(sequences):
                for index, x in enumerate(seq):
                    if i + 1 == x:
                        first_host[j, i] = index + 1
                        break

        _df = pd.DataFrame(first_host, columns=self.columns)
        if self.save:
            _df.to_csv(self.name, index=False)

        return _df


class LastTransformer(Transformer):
    def __init__(self, max_index, save=True) -> None:
        super().__init__('last.csv')
        self.max_index = max_index
        self.columns = [f'last_{i+1}' for i in range(max_index)]
        self.save = save

    def transform(self, sequences):
        last_host = np.zeros((len(sequences), self.max_index))
        for i in trange(self.max_index):
            for j, seq in enumerate(sequences):
                for index, x in enumerate(seq):
                    if i + 1 == x:
                        last_host[j, i] = index + 1

        _df = pd.DataFrame(last_host, columns=self.columns)
        if self.save:
            _df.to_csv(self.name, index=False)

        return _df


class MinDistanceTransformer(Transformer):
    def __init__(self, max_index, save=True) -> None:
        super().__init__('min_distance.csv')
        self.max_index = max_index
        self.columns = [f'min_dist_{i+1}' for i in range(max_index)]
        self.save = save

    def transform(self, sequences):
        min_dist_host = np.zeros((len(sequences), self.max_index))
        for i in trange(self.max_index):
            for j, seq in enumerate(sequences):
                indexes = [index for index, x in enumerate(seq) if i + 1 == x]
                if len(indexes) > 1:
                    distance = [indexes[i+1] - indexes[i]
                                for i in range(len(indexes) - 1)]
                    min_dist_host[j, i] = min(distance)

        _df = pd.DataFrame(min_dist_host, columns=self.columns)
        if self.save:
            _df.to_csv(self.name, index=False)

        return _df


class MaxDistanceTransformer(Transformer):
    def __init__(self, max_index, save=True) -> None:
        super().__init__('max_distance.csv')
        self.max_index = max_index
        self.columns = [f'max_dist_{i+1}' for i in range(max_index)]
        self.save = save

    def transform(self, sequences):
        max_dist_host = np.zeros((len(sequences), self.max_index))
        for i in trange(self.max_index):
            for j, seq in enumerate(sequences):
                indexes = [index for index, x in enumerate(seq) if i + 1 == x]
                if len(indexes) > 1:
                    distance = [
                        indexes[i+1] - indexes[i]
                        for i in range(len(indexes) - 1)
                    ]
                    max_dist_host[j, i] = max(distance)

        _df = pd.DataFrame(max_dist_host, columns=self.columns)
        if self.save:
            _df.to_csv(self.name, index=False)

        return _df


class LeftTransformer(Transformer):
    def __init__(self, max_index, save=True) -> None:
        super().__init__('left.csv')
        self.max_index = max_index
        self.columns = [f'left_{i+1}' for i in range(max_index)]
        self.save = save

    def transform(self, sequences):
        left_mode_host = np.zeros((len(sequences), self.max_index))
        for i in trange(self.max_index):
            for j, seq in enumerate(sequences):
                neibord = [
                    seq[index + 1]
                    for index in range(len(seq) - 1)
                    if i + 1 == seq[index]
                ]
                if len(neibord) > 0:
                    left_mode_host[j, i] = stats.mode(neibord)

        _df = pd.DataFrame(left_mode_host, columns=self.columns)
        if self.save:
            _df.to_csv(self.name, index=False)

        return _df


class RightTransformer(Transformer):
    def __init__(self, max_index, save=True) -> None:
        super().__init__('right.csv')
        self.max_index = max_index
        self.columns = [f'right_{i+1}' for i in range(max_index)]
        self.save = save

    def transform(self, sequences):
        right_mode_host = np.zeros((len(sequences), self.max_index))
        for i in trange(self.max_index):
            for j, seq in enumerate(sequences):
                neibord = [
                    seq[index - 1]
                    for index in range(1, len(seq))
                    if i + 1 == seq[index]
                ]
                if len(neibord) > 0:
                    right_mode_host[j, i] = stats.mode(neibord)

        _df = pd.DataFrame(right_mode_host, columns=self.columns)
        if self.save:
            _df.to_csv(self.name, index=False)

        return _df


class NumericalPCA(Transformer):
    def __init__(self, importance=0.85, scale=True) -> None:
        self.importance = importance
        self.scale = scale

    def model(self, n_components):
        if self.scale:
            return make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components)
            )

        return make_pipeline(PCA(n_components=n_components))

    def fit(self, X):
        _, n = X.shape
        pca_pipe = self.model(n)
        pca_pipe.fit(X)
        modelo_pca = pca_pipe.named_steps['pca']

        self.explained_accumulated = [0]
        self.n_components = None
        for i, percent in enumerate(modelo_pca.explained_variance_ratio_):
            self.explained_accumulated.append(
                percent + self.explained_accumulated[-1])

            if (
                not self.n_components and
                self.explained_accumulated[-1] > self.importance
            ):
                self.n_components = i + 1

        return None

    def transform(self, X):
        return self.model(self.n_components).fit_transform(X)

    def plot(self):
        _, ax = plt.subplots(figsize=(5, 5))

        plt.axvline(x=self.n_components, color='b', label='N Components')
        ax.plot(
            list(map(lambda x: x+1, range(len(self.explained_accumulated)))),
            self.explained_accumulated
        )
        ax.set_title('Porcentaje de varianza explicada por cada componente')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Por. varianza explicada')


class CategorialPCA(Transformer):
    def __init__(self, importance='0.90%') -> None:
        self.importance = importance

    def model(self, n_components):
        return prince.MCA(n_components=n_components)

    def fit(self, X):
        _, n = X.shape
        model = self.model(n)
        model = model.fit(X)
        variance = (
            model.eigenvalues_summary['% of variance (cumulative)']
            < self.importance
        )

        self.n_components = np.sum(variance) + 1
        return None

    def transform(self, X):
        return self.model(self.n_components).fit_transform(X).to_numpy()
