import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, Constant

from automl.components.base import PreprocessingAlgorithm


class KNNImputerComponent(PreprocessingAlgorithm):
    def __init__(self, missing_values=np.nan, n_neighbors: int = 5, weights: str = "uniform",
                 metric: str = "nan_euclidean", add_indicator: bool = False):
        super().__init__()

        # TODO what about missing values in categorical data?
        self.n_neighbors = n_neighbors
        self.missing_values = missing_values
        self.weights = weights
        self.metric = metric
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        from sklearn.impute import KNNImputer

        self.preprocessor = KNNImputer(missing_values=self.missing_values, n_neighbors=self.n_neighbors,
                                       weights=self.weights, metric=self.metric, add_indicator=self.add_indicator,
                                       copy=False)
        self.preprocessor = self.preprocessor.fit(X)
        return self



    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                'handles_missing_values': True,
                'handles_nominal_values': True,
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
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_neighbors = UniformIntegerHyperparameter("n_neighbors", 2, 50, default_value=5)
        weights = CategoricalHyperparameter("weights", ["uniform", "distance"], default_value="uniform")
        metric = Constant("metric", "nan_euclidean")
        add_indicator = CategoricalHyperparameter("add_indicator", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_neighbors, weights, metric, add_indicator])
        return cs
