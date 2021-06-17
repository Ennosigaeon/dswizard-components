import numpy as np
import pandas as pd

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class MultiColumnLabelEncoderComponent(PreprocessingAlgorithm):
    """MultiColumnLabelEncoderComponent

    A ColumnEncoder that can handle missing values and multiple categorical columns.
    Read more in the :ref:`User Guide`.

    Parameters
    ----------
    columns : List[str] (optional)
        List of column to be encoded

    Attributes
    ----------
    estimator_ : LabelEncoder
        The used LabelEncoder

    See also
    --------
    LabelEncoder

    References
    ----------
    """

    def __init__(self,
                 columns=None):
        super().__init__('multi_column_label_encoder')
        self.columns = columns
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
        categorical = df.select_dtypes(include=['category', 'object']).columns
        if len(categorical) == 0:
            return df.to_numpy()
        else:
            for colname in categorical:
                missing_vec = pd.isna(df[colname])
                df[colname] = df[colname].astype('category')
                df[colname].cat.add_categories(['<MISSING>'], inplace=True)
                df.loc[missing_vec, colname] = '<MISSING>'

                df[colname] = self.estimator_.fit_transform(df[colname].astype(str))
                df.loc[missing_vec, colname] = np.nan

        return df.to_numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'MultiColumnLabelEncoder',
                'name': 'MultiColumnLabelEncoder',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
