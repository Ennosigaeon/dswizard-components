from typing import List

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class BinarizerComponent(PreprocessingAlgorithm):

    def __init__(self, threshold: float = 0.0):
        super().__init__('binarizer')
        self.threshold = threshold

    def fit(self, X, y=None):
        self.estimator_ = self.to_sklearn(X.shape[0], X.shape[1], np.mean(np.var(X)))
        self.estimator_ = self.estimator_.fit(X)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, variance: float = 1, **kwargs):
        from sklearn.preprocessing import Binarizer

        # threshold = max(0., int(np.round(variance * self.threshold_factor, 0)))
        return Binarizer(threshold=self.threshold)

    def get_feature_names_out(self, input_features: List[str] = None):
        from sklearn.utils.validation import _check_feature_names_in
        return _check_feature_names_in(self, input_features)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'Binarizer',
                'name': 'Binarizer',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
