from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import resolve_factor, HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, \
    HANDLES_NUMERIC, HANDLES_MULTICLASS


class FactorAnalysisComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components_factor: float = None,
                 max_iter: int = 1000,
                 random_state=None,
                 svd_method: str = "randomized",
                 iterated_power: int = 3,
                 tol: float = 1e-2):
        super().__init__()
        self.n_components_factor = n_components_factor
        self.svd_method = svd_method
        self.iterated_power = iterated_power
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.decomposition import FactorAnalysis

        n_components = resolve_factor(self.n_components_factor, n_features, cs_default=1.)
        return FactorAnalysis(n_components=n_components,
                              svd_method=self.svd_method,
                              max_iter=self.max_iter,
                              iterated_power=self.iterated_power,
                              tol=self.tol,
                              random_state=self.random_state,
                              copy=False)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter('n_components_factor', 1, 250, default_value=10)
        max_iter = UniformIntegerHyperparameter("max_iter", 10, 2000, default_value=1000)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-2, log=True)
        svd_method = CategoricalHyperparameter("svd_method", ["lapack", "randomized"], default_value="randomized")
        iterated_power = UniformIntegerHyperparameter("iterated_power", 1, 10, default_value=3)
        cs.add_hyperparameters([n_components, max_iter, tol, svd_method, iterated_power])

        iterated_power_condition = InCondition(iterated_power, svd_method, ["randomized"])
        cs.add_condition(iterated_power_condition)

        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'FA',
                'name': 'Factor Analysis',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
