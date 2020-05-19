import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm


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
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        threshold_factor = UniformFloatHyperparameter("threshold_factor", 0., 1., default_value=0.)
        cs.add_hyperparameter(threshold_factor)
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        # TODO find out of this is right!
        return {'shortname': 'Binarizer',
                'name': 'Binarizer',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}
