from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS, resolve_factor


class TruncatedSVDComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components_factor: float = 0.5,
                 algorithm: str = 'randomized',
                 n_iter: int = 5,
                 tol: float = 0.,
                 random_state: int = None):
        super().__init__()
        self.n_components_factor = n_components_factor
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.decomposition import TruncatedSVD

        n_components = min(resolve_factor(self.n_components_factor,
                                          min(n_samples, n_features)), min(n_samples, n_features) - 1)
        return TruncatedSVD(n_components=n_components,
                            algorithm=self.algorithm,
                            n_iter=self.n_iter,
                            tol=self.tol,
                            random_state=self.random_state)

    @staticmethod
    def get_properties():
        return {'shortname': 'TSVD',
                'name': 'Truncated Singular Value Decomposition',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        n_components_factor = UniformFloatHyperparameter(name="n_components_factor", lower=0., upper=1.,
                                                         default_value=0.5)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components_factor])
        return cs
