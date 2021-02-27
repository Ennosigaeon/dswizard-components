import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, Constant

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS


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

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.impute import KNNImputer

        return KNNImputer(missing_values=self.missing_values, n_neighbors=self.n_neighbors,
                          weights=self.weights, metric=self.metric, add_indicator=self.add_indicator,
                          copy=False)

    @staticmethod
    def get_properties():
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        n_neighbors = UniformIntegerHyperparameter("n_neighbors", 2, 50, default_value=5)
        weights = CategoricalHyperparameter("weights", ["uniform", "distance"], default_value="uniform")
        metric = Constant("metric", "nan_euclidean")
        add_indicator = CategoricalHyperparameter("add_indicator", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_neighbors, weights, metric, add_indicator])
        return cs
