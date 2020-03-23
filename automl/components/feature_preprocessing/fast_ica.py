from scipy import sparse
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm


class FastICAComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components: int = None,
                 algorithm: str = 'parallel',
                 whiten: bool = True,
                 fun: str = 'logcosh',
                 max_iter: int = 200,
                 random_state=None,
                 tol: float = 0.0001):
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.decomposition import FastICA
        self.preprocessor = FastICA(n_components=self.n_components,
                                    algorithm=self.algorithm,
                                    whiten=self.whiten,
                                    fun=self.fun,
                                    max_iter=self.max_iter,
                                    random_state=self.random_state,
                                    tol=self.tol)
        self.preprocessor.fit(X)
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_components = UniformIntegerHyperparameter("n_components", 10, 2500, default_value=100)
        algorithm = CategoricalHyperparameter("algorithm", ["parallel", "deflation"], default_value="parallel")
        whiten = CategoricalHyperparameter("whiten", [True, False], default_value=True)
        fun = CategoricalHyperparameter("fun", ["logcosh", "exp", "cube"], default_value="logcosh")
        max_iter = UniformIntegerHyperparameter("max_iter", 1, 1000, default_value=100)
        tol = UniformFloatHyperparameter("tol", 1e-7, 10., default_value=1.)

        cs.add_hyperparameters([n_components, algorithm, whiten, fun, max_iter, tol])
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FastICA',
                'name': 'Fast Independent Component Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, UNSIGNED_DATA)
                }
