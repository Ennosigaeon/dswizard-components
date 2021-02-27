from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter, Constant

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class GaussianProcessClassifier(PredictionAlgorithm):

    def __init__(self,
                 kernel: str = None,
                 optimizer: str = "fmin_l_bfgs_b",
                 n_restarts_optimizer: int = 0,
                 max_iter_predict: int = 100,
                 multi_class: str = "one_vs_rest",
                 random_state=None
                 ):
        super().__init__()
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.multi_class = multi_class
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.gaussian_process import GaussianProcessClassifier

        if self.kernel == "constant":
            from sklearn.gaussian_process.kernels import ConstantKernel
            self.kernel = ConstantKernel()
        elif self.kernel == "rbf":
            from sklearn.gaussian_process.kernels import RBF
            self.kernel = RBF()
        elif self.kernel == "matern":
            from sklearn.gaussian_process.kernels import Matern
            self.kernel = Matern()
        elif self.kernel == "rational_quadratic":
            from sklearn.gaussian_process.kernels import RationalQuadratic
            self.kernel = RationalQuadratic()
        elif self.kernel == "exp_sin_squared":
            from sklearn.gaussian_process.kernels import ExpSineSquared
            self.kernel = ExpSineSquared()
        elif self.kernel == "white":
            from sklearn.gaussian_process.kernels import WhiteKernel
            self.kernel = WhiteKernel()
        elif self.kernel == "dot":
            from sklearn.gaussian_process.kernels import DotProduct
            self.kernel = DotProduct()

        return GaussianProcessClassifier(
            kernel=self.kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            multi_class=self.multi_class,
            random_state=self.random_state
        )

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'GP',
                'name': 'Gaussian Process Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        kernel = CategoricalHyperparameter("kernel",
                                           ["constant", "rbf", "matern", "rational_quadratic", "exp_sine_squared",
                                            "white", "dot"], default_value="rbf")
        optimizer = Constant("optimizer", "fmin_l_bfgs_b")
        n_restarts_optimizer = UniformIntegerHyperparameter("n_restarts_optimizer", 0, 500, default_value=0)
        max_iter_predict = UniformIntegerHyperparameter("max_iter_predict", 1, 1000, default_value=100)
        multi_class = CategoricalHyperparameter("multi_class", ["one_vs_rest", "one_vs_one"],
                                                default_value="one_vs_rest")

        cs.add_hyperparameters([n_restarts_optimizer, max_iter_predict, multi_class, kernel, optimizer])

        return cs
