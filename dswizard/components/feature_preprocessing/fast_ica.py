from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, \
    HANDLES_NUMERIC, HANDLES_MULTICLASS, resolve_factor


class FastICAComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components_factor: float = None,
                 algorithm: str = 'parallel',
                 whiten: bool = True,
                 fun: str = 'logcosh',
                 max_iter: int = 200,
                 random_state=None,
                 tol: float = 0.0001):
        super().__init__()
        self.n_components_factor = n_components_factor
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.decomposition import FastICA

        n_components = resolve_factor(self.n_components_factor, min(n_samples, n_features), cs_default=1.)
        return FastICA(n_components=n_components,
                       algorithm=self.algorithm,
                       whiten=self.whiten,
                       fun=self.fun,
                       max_iter=self.max_iter,
                       random_state=self.random_state,
                       tol=self.tol)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        n_components_factor = UniformFloatHyperparameter("n_components_factor", 0., 1., default_value=1.)
        algorithm = CategoricalHyperparameter("algorithm", ["parallel", "deflation"], default_value="parallel")
        whiten = CategoricalHyperparameter("whiten", [True, False], default_value=True)
        fun = CategoricalHyperparameter("fun", ["logcosh", "exp", "cube"], default_value="logcosh")
        cs.add_hyperparameters([n_components_factor, algorithm, whiten, fun])

        cs.add_condition(EqualsCondition(n_components_factor, whiten, True))
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'FastICA',
                'name': 'Fast Independent Component Analysis',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
