from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm


class BernoulliRBM(PreprocessingAlgorithm):
    def __init__(self, n_components: int = 256, learning_rate: float = 0.1, batch_size: int = 10, n_iter: int = 10,
                 random_state=None):
        super().__init__()
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.neural_network import BernoulliRBM

        self.preprocessor = BernoulliRBM(n_components=self.n_components, learning_rate=self.learning_rate,
                                         batch_size=self.batch_size, n_iter=self.n_iter, random_state=self.random_state)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_components = UniformIntegerHyperparameter("n_components", 1, 2000, default_value=256)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-5, 15, default_value=0.1)
        batch_size = UniformIntegerHyperparameter("batch_size", 1, 100, default_value=10)
        n_iter = UniformIntegerHyperparameter("n_iter", 2, 2000, default_value=10)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, n_iter, learning_rate, batch_size])
        return cs
