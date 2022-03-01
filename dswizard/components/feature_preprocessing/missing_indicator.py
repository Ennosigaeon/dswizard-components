from typing import List

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class MissingIndicatorComponent(PreprocessingAlgorithm):
    def __init__(self, missing_values=np.nan, features: str = "all"):
        super().__init__('missing_indicator')
        try:
            if np.isnan(missing_values):
                self.args['missing_values'] = 'NaN'
        except TypeError:
            pass

        self.features = features
        self.missing_values = missing_values

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.impute import MissingIndicator
        return MissingIndicator(missing_values=self.missing_values, features=self.features)

    def transform(self, X):
        if self.estimator_ is None:
            raise ValueError()
        missing = self.estimator_.transform(X)
        return np.any(missing, axis=1, keepdims=True).astype(int)

    def get_feature_names_out(self, input_features: List[str] = None):
        return np.array(['missing_values'])

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
    def deserialize(**kwargs) -> 'MissingIndicatorComponent':
        if 'missing_values' in kwargs and kwargs['missing_values'] == 'NaN':
            kwargs['missing_values'] = np.nan
        return MissingIndicatorComponent(**kwargs)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        return cs
