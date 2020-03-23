import numpy as np
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm


class PCAComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components: float = None,
                 whiten: bool = False,
                 svd_solver: str = "auto",
                 tol: float = 0.,
                 iterated_power: int = "auto",
                 random_state=None):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, Y=None):
        num_features = X.shape[1]
        if self.n_components is None:
            n_components = None
        else:
            n_components = max(1, int(np.round(self.n_components * num_features, 0)))
        from sklearn.decomposition import PCA
        self.preprocessor = PCA(n_components=n_components,
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
        keep_variance = UniformFloatHyperparameter("n_components", 0., 1., default_value=0.9999)
        whiten = CategoricalHyperparameter("whiten", [False, True], default_value=False)
        svd_solver = CategoricalHyperparameter("svd_solver", ["full", "arpack", "randomized"], default_value="full")
        tol = UniformFloatHyperparameter("tol", 0., 5., default_value=1e-2)
        iterated_power = UniformIntegerHyperparameter("iterated_power", 0, 1000, default_value=1000)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, whiten, svd_solver, tol, iterated_power])

        iterated_power_condition = EqualsCondition(iterated_power, svd_solver, "randomized")
        cs.add_condition(iterated_power_condition)
        return cs
