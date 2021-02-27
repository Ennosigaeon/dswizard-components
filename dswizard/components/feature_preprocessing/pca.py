import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import resolve_factor, HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, \
    HANDLES_NUMERIC, HANDLES_MULTICLASS


class PCAComponent(PreprocessingAlgorithm):
    def __init__(self,
                 keep_variance: float = None,
                 whiten: bool = False,
                 svd_solver: str = "auto",
                 tol: float = 0.,
                 iterated_power: int = "auto",
                 random_state=None):
        super().__init__()
        self.keep_variance = keep_variance
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, Y=None):
        self.preprocessor = self.to_sklearn(X.shape[0], X.shape[1])
        self.preprocessor.fit(X)

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.decomposition import PCA
        n_components = resolve_factor(self.keep_variance, min(n_samples, n_features), cs_default=0.9999)
        return PCA(n_components=n_components,
                   whiten=self.whiten,
                   random_state=self.random_state,
                   svd_solver=self.svd_solver,
                   tol=self.tol,
                   iterated_power=self.iterated_power,
                   copy=False)

    @staticmethod
    def get_properties():
        return {'shortname': 'PCA',
                'name': 'Principle Component Analysis',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        keep_variance = UniformFloatHyperparameter("keep_variance", 0.5, 0.9999, default_value=0.9999)
        whiten = CategoricalHyperparameter("whiten", [False, True], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, whiten])
        return cs
