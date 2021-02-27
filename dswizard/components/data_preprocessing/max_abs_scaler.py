from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class MaxAbsScalerComponent(PreprocessingAlgorithm):

    def __init__(self):
        super().__init__()

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import MaxAbsScaler
        return MaxAbsScaler(copy=False)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'MaxAbsScaler',
                'name': 'MaxAbsScaler',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
