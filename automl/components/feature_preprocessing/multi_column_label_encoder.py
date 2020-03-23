import numpy as np
import pandas as pd

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

        categorical = {}

        for i in range(X.shape[1]):
            try:
                X.iloc[:, i].values.astype(float)
                categorical[X.columns[i]] = False
            except ValueError:
                categorical[X.columns[i]] = True
        if not np.any(categorical.values()):
            return X.to_numpy()
        else:
            for colname, col in X.iteritems():
                if categorical[colname]:
                    X[colname] = self.preprocessor.fit_transform(col)
        return X.to_numpy()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MultiColumnLabelEncoder',
                'name': 'Multi Column Label Encoder',
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
