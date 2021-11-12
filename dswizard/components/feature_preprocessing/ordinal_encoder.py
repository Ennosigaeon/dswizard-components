import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class OrdinalEncoderComponent(PreprocessingAlgorithm):
    """OrdinalEncoderComponent

    A ColumnEncoder that can handle missing values and multiple categorical columns.
    Read more in the :ref:`User Guide`.

    Attributes
    ----------
    estimator_ : OrdinalEncoder
        The used OrdinalEncoder

    See also
    --------
    OrdinalEncoder

    References
    ----------
    """

    def __init__(self):
        super().__init__('ordinal_encoder')
        from sklearn.preprocessing import LabelEncoder
        self.estimator_ = LabelEncoder()

    def fit(self, X, y=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OrdinalEncoder

        df = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1])).infer_objects()
        categorical = make_column_selector(dtype_exclude=np.number)

        self.estimator_ = ColumnTransformer(
            [('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical)],
            remainder='passthrough')
        self.estimator_.fit(df, y)
        return self

    @staticmethod
    def get_properties():
        return {'shortname': 'MultiColumnLabelEncoder',
                'name': 'MultiColumnLabelEncoder',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
