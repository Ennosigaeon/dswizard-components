import numpy as np
import pandas as pd

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
        return self  # not relevant here

    def transform(self, X: np.ndarray):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """

        df = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1]))
        categorical = df.select_dtypes(exclude=np.number).columns
        if len(categorical) == 0:
            return df.to_numpy()
        else:
            for colname in categorical:
                missing_vec = pd.isna(df[colname])
                df[colname] = df[colname].astype('category').cat.add_categories(['<MISSING>'])
                df.loc[missing_vec, colname] = '<MISSING>'

                df[colname] = self.estimator_.fit_transform(df[colname].astype(str))
                df.loc[missing_vec, colname] = np.nan

        return df.to_numpy()

    @staticmethod
    def get_properties():
        return {'shortname': 'MultiColumnLabelEncoder',
                'name': 'MultiColumnLabelEncoder',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
