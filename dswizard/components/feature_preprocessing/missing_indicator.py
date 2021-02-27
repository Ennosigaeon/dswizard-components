import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class MissingIndicatorComponent(PreprocessingAlgorithm):
    def __init__(self, missing_values=np.nan, features: str = "missing-only"):
        super().__init__()

        self.features = features
        self.missing_values = missing_values

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.impute import MissingIndicator
        return MissingIndicator(missing_values=self.missing_values, features=self.features)

    @staticmethod
    def get_properties():
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        features = CategoricalHyperparameter("features", ["missing-only", "all"], default_value="missing-only")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features])
        return cs
