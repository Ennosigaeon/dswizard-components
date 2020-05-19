import numpy as np
import pandas as pd

from automl.components.base import PreprocessingAlgorithm


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
        return self

    def transform(self, X: pd.DataFrame):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1]))

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
