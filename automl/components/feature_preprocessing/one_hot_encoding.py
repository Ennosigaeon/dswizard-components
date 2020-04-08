import numpy as np
import pandas as pd

from automl.components.base import PreprocessingAlgorithm


class OneHotEncoderComponent(PreprocessingAlgorithm):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        dummy_na = np.any(pd.isna(X))
        X = pd.get_dummies(X, sparse=False, dummy_na=dummy_na)
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
