from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm
import numpy as np

class BinarizerComponent(PreprocessingAlgorithm):

    def __init__(self, threshold_factor: float = 0.):
        super().__init__()
        self.threshold_factor = threshold_factor

    def fit(self, X, y=None):
        from sklearn.preprocessing import Binarizer

        variance = np.var(X)
        print(variance)
        threshold = max(1, int(np.round(variance[0] * self.threshold_factor, 0)))

        self.preprocessor = Binarizer(threshold=threshold)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        # TODO Use fraction of data per column
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
