from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class BernoulliRBM(PreprocessingAlgorithm):
    def __init__(self, n_components: int = 256, learning_rate: float = 0.1, batch_size: int = 10, n_iter: int = 10,
                 random_state=None):
        super().__init__()
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.neural_network import BernoulliRBM

        return BernoulliRBM(n_components=self.n_components, learning_rate=self.learning_rate,
                            batch_size=self.batch_size, n_iter=self.n_iter, random_state=self.random_state)

    @staticmethod
    def get_properties():
        return {'shortname': 'BernoulliRBM',
                'name': 'BernoulliRBM',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        n_components = UniformIntegerHyperparameter("n_components", 1, 512, default_value=256)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 1., default_value=0.1)
        batch_size = UniformIntegerHyperparameter("batch_size", 1, 100, default_value=10)
        n_iter = UniformIntegerHyperparameter("n_iter", 2, 200, default_value=10)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, n_iter, learning_rate, batch_size])
        return cs
