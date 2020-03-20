import warnings

import numpy as np
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm


class KernelPCAComponent(PreprocessingAlgorithm):
    def __init__(self, n_components: int = None,
                 kernel: str = 'linear',
                 degree: int = 3,
                 gamma: float = 0.25,
                 coef0: float = 0.0,
                 alpha: int = 1,
                 fit_inverse_transform: bool = True,
                 eigen_solver: str = "dense",
                 tol: float = 0.,
                 max_iter: int = 10000,
                 remove_zero_eigen: bool = False,
                 copy_X: bool = True,
                 random_state=None):
        super().__init__()
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eigen = remove_zero_eigen
        self.copy_X = copy_X
        self.random_state = random_state

    def fit(self, X, Y=None):
        import scipy.sparse
        from sklearn.decomposition import KernelPCA

        if self.n_components is not None:
            self.n_components = int(self.n_components)
        self.degree = int(self.degree)
        self.gamma = float(self.gamma)
        self.coef0 = float(self.coef0)

        self.preprocessor = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            random_state=self.random_state,
            alpha=self.alpha,
            fit_inverse_transform=self.fit_inverse_transform,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            remove_zero_eig=self.remove_zero_eigen,
            copy_X=self.copy_X)

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
    def get_properties(dataset_properties=None):
        return {'shortname': 'KernelPCA',
                'name': 'Kernel Principal Component Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (DENSE, UNSIGNED_DATA)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_components = UniformIntegerHyperparameter("n_components", 10, 3000, default_value=100)
        kernel = CategoricalHyperparameter('kernel', ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
        gamma = UniformFloatHyperparameter("gamma", 1e-09, 15., log=True, default_value=1.0)
        degree = UniformIntegerHyperparameter('degree', 2, 6, 3)
        coef0 = UniformFloatHyperparameter("coef0", -1., 1., default_value=0.)
        alpha = UniformIntegerHyperparameter("alpha", 1e-9, 5., default_value=1.)
        fit_inverse_transform = CategoricalHyperparameter("fit_inverse_transform", [True, False], default_value=False)
        eigen_solver = CategoricalHyperparameter("eigen_solver", ["dense", "arpack"], default_value="dense")
        tol = UniformFloatHyperparameter("tol", 0., 2., default_value=0.)
        max_iter = UniformIntegerHyperparameter("max_iter", 1, 1000, default_value=100)
        remove_zero_eigen = CategoricalHyperparameter("remove_zero_eigen", [True, False], default_value=False)
        copy_X = CategoricalHyperparameter("copy_X", [True, False], default_value=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [n_components, kernel, degree, gamma, coef0, alpha, fit_inverse_transform, eigen_solver, tol, max_iter,
             remove_zero_eigen, copy_X])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
        alpha_condition = EqualsCondition(alpha, fit_inverse_transform, True)
        tol_condition = InCondition(tol, eigen_solver, ["arpack"])
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition, alpha_condition, tol_condition])
        return cs
