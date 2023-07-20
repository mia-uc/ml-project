from sklearn.pipeline import make_pipeline
from .transformers import NumericalPCA
import pandas as pd
from .clusters import BestKMeans


def numerical_pipelines(
    label, X, y, transformer,
    pca_p=.9, k_min=20, k_max=100,
    d_u=None, d_v=None
):

    df = transformer.fit_transform(X)
    pca_data = NumericalPCA(.9).fit_transform(df)
    m, n = pca_data.shape
    pca_df = pd.DataFrame(
        pca_data,
        columns=[f'{label}_PCA_{i + 1}' for i in range(n)]
    )

    kmeans_labels = BestKMeans(pca_data, y, k_min, k_max)

    _df = pd.concat([df, pca_df], axis=1)
    _df.to_csv(label, index=False)
    return _df
