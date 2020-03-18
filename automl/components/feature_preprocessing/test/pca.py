import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import check_for_bool


class PCAComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components: float = None,
                 whiten: bool = False,
                 svd_solver: str = "full",
                 tol: float = 1e-2,
                 iterated_power: int = 1000,
                 random_state=None):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, Y=None):
        from sklearn.decomposition import PCA
        self.whiten = check_for_bool(self.whiten)

        if self.svd_solver is "randomized":
            self.n_components = int(len(X) * self.n_components)

        self.preprocessor = PCA(n_components=self.n_components,
                                whiten=self.whiten,
                                random_state=self.random_state,
                                svd_solver=self.svd_solver,
                                tol=self.tol,
                                iterated_power=self.iterated_power,
                                copy=False)
        self.preprocessor.fit(X)

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PCA',
                'name': 'Principle Component Analysis',
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
        whiten = CategoricalHyperparameter("whiten", [False, True], default_value=False)
        svd_solver = CategoricalHyperparameter("svd_solver", ["full","arpack","randomized"], default_value="full")
        tol = UniformFloatHyperparameter("tol", 0., 5., default_value= 1e-2)
        iterated_power = UniformIntegerHyperparameter("iterated_power", 0, 10000, default_value=1000)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, whiten, svd_solver, tol, iterated_power])

        iterated_power_condition = EqualsCondition(iterated_power, svd_solver, "randomized")
        cs.add_condition(iterated_power_condition)
        return cs
