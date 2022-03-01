from typing import List

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS


class KNNImputerComponent(PreprocessingAlgorithm):
    def __init__(self, missing_values=np.nan, n_neighbors: int = 5, weights: str = "uniform",
                 metric: str = "nan_euclidean", add_indicator: bool = False):
        super().__init__('knn_imputer')
        try:
            if np.isnan(missing_values):
                self.args['missing_values'] = 'NaN'
        except TypeError:
            pass

        # TODO what about missing values in categorical data?
        self.n_neighbors = n_neighbors
        self.missing_values = missing_values
        self.weights = weights
        self.metric = metric
        self.add_indicator = add_indicator

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.impute import KNNImputer

        return KNNImputer(missing_values=self.missing_values, n_neighbors=self.n_neighbors,
                          weights=self.weights, metric=self.metric, add_indicator=self.add_indicator)

    def get_feature_names_out(self, input_features: List[str] = None):
        from sklearn.utils.validation import _check_feature_names_in
        names = _check_feature_names_in(self, input_features)

        if self.add_indicator:
            columns = [f'missing_{input_features[idx]}' for idx in self.estimator_.indicator_.features_]
            names = np.append(names, columns)

        return names

    @staticmethod
    def get_properties():
        return {'shortname': 'KNN Imputation',
                'name': 'KNN Imputation',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def deserialize(**kwargs) -> 'KNNImputerComponent':
        if 'missing_values' in kwargs and kwargs['missing_values'] == 'NaN':
            kwargs['missing_values'] = np.nan
        return KNNImputerComponent(**kwargs)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        n_neighbors = UniformIntegerHyperparameter("n_neighbors", 2, 50, default_value=5)
        weights = CategoricalHyperparameter("weights", ["uniform", "distance"], default_value="uniform")
        add_indicator = CategoricalHyperparameter("add_indicator", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_neighbors, weights, add_indicator])
        return cs
