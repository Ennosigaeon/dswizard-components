from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm
from util.common import resolve_factor


class FactorAnalysisComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components_factor: float = None,
                 max_iter: int = 1000,
                 random_state=None,
                 svd_method: str = "randomized",
                 iterated_power: int = 3,
                 tol: float = 1e-2,
                 copy=False):
        super().__init__()
        self.n_components_factor = n_components_factor
        self.svd_method = svd_method
        self.iterated_power = iterated_power
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.copy = copy

    def fit(self, X, y=None):
        from sklearn.decomposition import FactorAnalysis

        n_components = resolve_factor(self.n_components_factor, X.shape[1])
        self.preprocessor = FactorAnalysis(n_components=n_components,
                                           svd_method=self.svd_method,
                                           max_iter=self.max_iter,
                                           iterated_power=self.iterated_power,
                                           tol=self.tol,
                                           random_state=self.random_state,
                                           copy=self.copy)
        self.preprocessor.fit(X)
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_components_factor = UniformFloatHyperparameter("n_components_factor", 0., 1., default_value=1.)
        max_iter = UniformIntegerHyperparameter("max_iter", 1, 10000, default_value=1000)
        tol = UniformFloatHyperparameter("tol", 1e-5, 5, default_value=1e-2)
        svd_method = CategoricalHyperparameter("svd_method", ["lapack", "randomized"], default_value="randomized")
        iterated_power = UniformIntegerHyperparameter("iterated_power", 1, 50, default_value=3)

        cs.add_hyperparameters([n_components_factor, max_iter, tol, svd_method, iterated_power])

        iterated_power_condition = InCondition(iterated_power, svd_method, ["randomized"])
        cs.add_condition(iterated_power_condition)

        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FA',
                'name': 'Factor Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, UNSIGNED_DATA)
                }
