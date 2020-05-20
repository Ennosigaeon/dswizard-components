import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class BinarizerComponent(PreprocessingAlgorithm):

    def __init__(self, threshold_factor: float = 0.0):
        super().__init__()
        self.threshold_factor = threshold_factor

    def fit(self, X, y=None):
        self.preprocessor = self.to_sklearn(X.shape[0], X.shape[1], np.mean(np.var(X)))
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, variance: float = 1, **kwargs):
        from sklearn.preprocessing import Binarizer

        threshold = max(0., int(np.round(variance * self.threshold_factor, 0)))
        return Binarizer(threshold=threshold, copy=False)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        threshold_factor = UniformFloatHyperparameter("threshold_factor", 0., 1., default_value=0.)
        cs.add_hyperparameter(threshold_factor)
        return cs

    @staticmethod
    def get_properties():
        # TODO find out of this is right!
        return {'shortname': 'Binarizer',
                'name': 'Binarizer',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
