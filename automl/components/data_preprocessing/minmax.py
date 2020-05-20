from ConfigSpace.configuration_space import ConfigurationSpace

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, HANDLES_NOMINAL_CLASS


class MinMaxScalerComponent(PreprocessingAlgorithm):

    def __init__(self):
        super().__init__()

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler(copy=False)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        # Feature Range?
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'MinMaxScaler',
                'name': 'MinMaxScaler',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
