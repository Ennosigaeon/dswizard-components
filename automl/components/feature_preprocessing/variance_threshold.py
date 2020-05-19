from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant

from automl.components.base import PreprocessingAlgorithm


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
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Variance Threshold',
            'name': 'Variance Threshold (constant feature removal)',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'is_deterministic': True,
            'handles_sparse': True,
            'handles_dense': True,
            # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
            # 'output': (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        # TODO missing
        threshold = Constant('threshold', 0.)
        cs.add_hyperparameter(threshold)
        return cs
