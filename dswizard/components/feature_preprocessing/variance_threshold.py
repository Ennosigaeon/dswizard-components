from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class VarianceThresholdComponent(PreprocessingAlgorithm):
    """Feature selector that removes all low-variance features.
    Features with a training-set variance lower than this threshold will be removed. The default is to keep all
    features with non-zero variance, i.e. remove the features that have the same value in all samples."""

    def __init__(self, threshold: float = 0.):
        super().__init__()
        self.threshold = threshold

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.feature_selection import VarianceThreshold
        return VarianceThreshold(threshold=self.threshold)

    @staticmethod
    def get_properties():
        return {'shortname': 'Variance Threshold',
                'name': 'Variance Threshold (constant feature removal)',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        threshold = UniformFloatHyperparameter("threshold", 0.0001, 0.2, default_value=0.0001)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(threshold)
        return cs
