from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from automl.components.base import PreprocessingAlgorithm
from util.common import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, HANDLES_MULTICLASS


class StandardScalerComponent(PreprocessingAlgorithm):

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import StandardScaler
        return StandardScaler(with_std=self.with_std, with_mean=self.with_mean, copy=False)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        with_mean = CategoricalHyperparameter("with_mean", [True, False], default_value=True)
        with_std = CategoricalHyperparameter("with_std", [True, False], default_value=True)

        cs.add_hyperparameters([with_mean, with_std])
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'StandardScaler',
                'name': 'StandardScaler',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
