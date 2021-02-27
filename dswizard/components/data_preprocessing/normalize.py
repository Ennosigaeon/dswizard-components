from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class NormalizerComponent(PreprocessingAlgorithm):
    def __init__(self, norm: str = 'l2'):
        super().__init__()
        self.norm = norm

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import Normalizer
        return Normalizer(norm=self.norm, copy=False)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'Normalizer',
                'name': 'Normalizer',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
