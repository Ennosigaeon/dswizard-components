import numpy as np
import pandas as pd

from automl.components.base import PreprocessingAlgorithm


class OneHotEncoderComponent(PreprocessingAlgorithm):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):

        dummy_na = False
        # categorical = {}
        # for i in range(X.shape[1]):
        #     try:
        #         X.iloc[:, i].values.astype(float)
        #         categorical[X.columns[i]] = False
        #     except ValueError:
        #         categorical[X.columns[i]] = True
        #         cat.append(X.columns[i])
        # if not np.any(categorical.values()):
        #     return X.to_numpy()
        cat = []

        if np.any(pd.isna(X)):
            dummy_na = True

        X_object = X.select_dtypes(include=['category', 'object'])

        for i in range(X_object.shape[1]):
            cat.append(X_object.columns[i])

        if len(cat) == 0:
            return X.to_numpy()

        X = pd.get_dummies(X, prefix=[cat[i] for i in range(len(cat))], sparse=False, dummy_na=dummy_na)

        return X.to_numpy()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True, }
        # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
        # 'output': (INPUT,), }
