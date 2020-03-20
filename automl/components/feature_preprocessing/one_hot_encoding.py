import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm


class OneHotEncoderComponent(PreprocessingAlgorithm):
    def __init__(self, categories: str = 'auto', sparse: bool = False, drop: str = None):
        super().__init__()
        self.sparse = sparse
        self.drop = drop
        self.categories = categories

    def fit(self, X, y=None):
        from sklearn.preprocessing import OneHotEncoder
        self.preprocessor = OneHotEncoder(categories=self.categories, sparse=self.sparse, drop=self.drop)

        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        # TODO OHE can not handle missing values

        categorical = {}

        for i in range(X.shape[1]):
            try:
                X.iloc[:, i].values.astype(float)
                categorical[X.columns[i]] = False
            except ValueError:
                categorical[X.columns[i]] = True

        if not np.any(categorical.values()):
            return X.to_numpy()

        for colname, col in X.iteritems():
            if categorical[colname]:
               X = pd.get_dummies(X, prefix=colname)

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

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        drop = CategoricalHyperparameter("drop", [None, "first"], default_value="first")
        cs.add_hyperparameters([drop])

        return cs
