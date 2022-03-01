from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector

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
        super().__init__('one_hot_encoding')

    def fit(self, X, y=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        df = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1])).infer_objects()
        categorical = make_column_selector(dtype_exclude=np.number)

        self.estimator_ = ColumnTransformer(
            [('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)],
            remainder='passthrough')
        self.estimator_.fit(df, y)
        return self

    def transform(self, X: np.ndarray):
        return self.estimator_.transform(X)

    def get_feature_names_out(self, input_features: List[str] = None):
        output_features = super().get_feature_names_out(input_features)
        return np.array([f.split('__')[-1] for f in output_features])

    @staticmethod
    def get_properties():
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
