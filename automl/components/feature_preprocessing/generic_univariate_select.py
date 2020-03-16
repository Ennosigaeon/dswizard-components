from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm


class GenericUnivariateSelectComponent(PreprocessingAlgorithm):

    """Feature selector that removes all low-variance features.
    Features with a training-set variance lower than this threshold will be removed. The default is to keep all
    features with non-zero variance, i.e. remove the features that have the same value in all samples."""

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        from sklearn.feature_selection import GenericUnivariateSelect
        self.preprocessor = GenericUnivariateSelect()
        self.preprocessor = self.preprocessor.fit(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'GenericUnivariateSelect',
            'name': 'Generic Univariate Select',
            # TODO Check if True
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

