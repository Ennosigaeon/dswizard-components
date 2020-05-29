import warnings

import numpy as np
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm

from automl.util.common import resolve_factor, HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL,\
    HANDLES_NUMERIC, HANDLES_MULTICLASS


class KernelPCAComponent(PreprocessingAlgorithm):
    def __init__(self, n_components_factor: int = None,
                 kernel: str = 'linear',
                 degree: int = 3,
                 gamma: float = None,
                 coef0: float = 1,
                 alpha: int = 1.,
                 fit_inverse_transform: bool = False,
                 eigen_solver: str = "auto",
                 tol: float = 0,
                 max_iter: int = None,
                 remove_zero_eig: bool = False,
                 random_state=None):
        super().__init__()
        self.n_components_factor = n_components_factor
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state

    def fit(self, X, Y=None):
        import scipy.sparse

        self.preprocessor = self.to_sklearn(X.shape[0], X.shape[1])
        if scipy.sparse.issparse(X):
            X = X.astype(np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            self.preprocessor.fit(X)

        # Raise an informative error message, equation is based ~line 249 in
        # kernel_pca.py in scikit-learn
        if len(self.preprocessor.alphas_ / self.preprocessor.lambdas_) == 0:
            raise ValueError('KernelPCA removed all features!')
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.decomposition import KernelPCA

        n_components = resolve_factor(self.n_components_factor, min(n_samples, n_features), cs_default=1.)

        gamma = None if self.gamma == 1. else self.gamma
        max_iter = None if self.max_iter == 100 else self.max_iter

        return KernelPCA(
            n_components=n_components,
            kernel=self.kernel,
            degree=self.degree,
            gamma=gamma,
            coef0=self.coef0,
            random_state=self.random_state,
            alpha=self.alpha,
            fit_inverse_transform=self.fit_inverse_transform,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=max_iter,
            remove_zero_eig=self.remove_zero_eig,
            n_jobs=1,
            copy_X=False)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            X_new = self.preprocessor.transform(X)

            # TODO write a unittest for this case
            if X_new.shape[1] == 0:
                raise ValueError("KernelPCA removed all features!")

            return X_new

    @staticmethod
    def get_properties():
        return {'shortname': 'KernelPCA',
                'name': 'Kernel Principal Component Analysis',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        n_components_factor = UniformFloatHyperparameter("n_components_factor", 0., 1., default_value=1.)
        kernel = CategoricalHyperparameter('kernel', ['poly', 'rbf', 'sigmoid', 'cosine', 'linear'], 'linear')
        gamma = UniformFloatHyperparameter("gamma", 1e-09, 15., log=True, default_value=1.0)
        degree = UniformIntegerHyperparameter('degree', 2, 6, 3)
        coef0 = UniformFloatHyperparameter("coef0", -10., 10., default_value=0.)
        alpha = UniformIntegerHyperparameter("alpha", 1e-9, 5., default_value=1.)
        fit_inverse_transform = CategoricalHyperparameter("fit_inverse_transform", [True, False], default_value=False)
        eigen_solver = CategoricalHyperparameter("eigen_solver", ["auto", "dense", "arpack"], default_value="auto")
        tol = UniformFloatHyperparameter("tol", 0., 2., default_value=0.)
        max_iter = UniformIntegerHyperparameter("max_iter", 1, 1000, default_value=100)
        remove_zero_eig = CategoricalHyperparameter("remove_zero_eig", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [n_components_factor, kernel, degree, gamma, coef0, alpha, fit_inverse_transform, eigen_solver, tol, max_iter,
             remove_zero_eig])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
        alpha_condition = EqualsCondition(alpha, fit_inverse_transform, True)
        tol_condition = InCondition(tol, eigen_solver, ["arpack"])
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition, alpha_condition, tol_condition])
        return cs
