from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from dswizard.components.base import NoopComponent
from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class OneHotEncoderComponent(PreprocessingAlgorithm):
    """OneHotEncoderComponent

    A OneHotEncoder that can handle missing values and multiple categorical columns.
    Read more in the :ref:`User Guide`.

    Parameters
    ----------

    Attributes
    ----------

    See also
    --------
    OneHotEncoder

    References
    ----------
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1])).infer_objects()
        else:
            df = X

        categorical = (df.dtypes == object).to_numpy()
        if not categorical.any():
            self.preprocessor = NoopComponent()
        else:
            self.preprocessor = self.to_sklearn(X.shape[0], X.shape[1], categorical=categorical)
            self.preprocessor.fit(X, y)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, categorical: List[bool] = 'auto', **kwargs):
        from sklearn.preprocessing import OneHotEncoder
        return ColumnTransformer([('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)],
                                 remainder='passthrough')
        # return OneHotEncoder(sparse=False, categories=categorical_columns, handle_unknown='ignore')

    def transform(self, X: pd.DataFrame):
        # if isinstance(X, np.ndarray):
        #     X = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1])).infer_objects()
        #
        # dummy_na = np.any(pd.isna(X))
        # X = pd.get_dummies(X, sparse=False, dummy_na=dummy_na)
        # return X.to_numpy()

        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
