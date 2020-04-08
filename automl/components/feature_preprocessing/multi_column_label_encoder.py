import pandas as pd
import numpy as np

from automl.components.base import PreprocessingAlgorithm


class MultiColumnLabelEncoderComponent(PreprocessingAlgorithm):

    def __init__(self,
                 columns=None):
        super().__init__()
        self.columns = columns
        from sklearn.preprocessing import LabelEncoder
        self.preprocessor = LabelEncoder()

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X: pd.DataFrame):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            return X.to_numpy()
        else:
            for colname, col in X.iteritems():
                if colname in X_object.columns:
                    missing_vec = pd.isna(X[colname])
                    missing_vec = missing_vec[missing_vec == True]
                    X[colname] = X[colname].cat.add_categories(['<MISSING>'])
                    X.loc[missing_vec.index, colname] = '<MISSING>'

                    X[colname] = self.preprocessor.fit_transform(col.astype(str))

                    X.loc[missing_vec.index, colname] = np.nan

        return X.to_numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MultiColumnLabelEncoder',
                'name': 'MultiColumnLabelEncoder',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}
