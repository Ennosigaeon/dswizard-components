import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import check_for_bool


class KernelPCAComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components: float = None,
                 kernel: str = 'linear',

                 random_state=None):
        super().__init__()
        self.n_components = n_components
        self.kernel = kernel

        self.random_state = random_state

    def fit(self, X, Y=None):
        from sklearn.decomposition import KernelPCA

        self.preprocessor = KernelPCA(n_components=self.n_components,
                                kernel=self.kernel,
                                random_state=self.random_state,
                                copy_X=False)
        self.preprocessor.fit(X)

        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KernelPCA',
                'name': 'Kernel Principal component analysis',
                # TODO Check if true
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO document that we have to be very careful
                'is_deterministic': False,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (DENSE, UNSIGNED_DATA)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        keep_variance = UniformFloatHyperparameter("n_components", 0.5, 0.9999, default_value=0.9999)
        kernel = CategoricalHyperparameter("kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'],
                                           default_value='linear')
        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, kernel])
        return cs
