import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace

from automl.components.base import PreprocessingAlgorithm


class OneHotEncoderComponent(PreprocessingAlgorithm):
    def __init__(self, categories: str = 'auto', sparse: bool = False):
        super().__init__()
        self.sparse = sparse
        self.categories = categories  # TODO testen ob es immernoch funktioniert wenn man OHE categories übergibt

    def fit(self, X, y=None):
        from sklearn.preprocessing import OneHotEncoder
        self.preprocessor = OneHotEncoder(categories=self.categories, sparse=self.sparse)

        self.preprocessor.fit(X)
        return self

    def transform(self, X: np.ndarray):

        categorical = []
        numeric = []

        for i in range(X.shape[1]):
            try:
                X.iloc[:, i].values.astype(float)
                numeric.append(i)
            except ValueError:
                categorical.append(i)

        if len(categorical) == 0:
            df = pd.DataFrame(X)
            return df.to_numpy()

        df = pd.DataFrame.from_records(X.iloc[:, categorical].values)
        cat = X.columns[categorical].values.tolist()
        df = pd.get_dummies(df, prefix=[cat[i] for i in range(len(cat))])

        for i in numeric:
            df[len(df.columns)] = X.iloc[:, i].values.astype(float)
        return df.to_numpy()

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

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
