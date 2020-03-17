import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class GaussianProcessClassifier(PredictionAlgorithm):

    def __init__(self,
                 kernel: str = "rbf",
                 optimizer: str = "fmin_l_bfgs_b",
                 n_restarts_optimizer: int = 0,
                 max_iter_predict: int = 100,
                 warm_start: bool = False,
                 copy_X_train: bool = True,
                 multi_class: str = "one_vs_rest",
                 ):
        super().__init__()
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.multi_class = multi_class

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessClassifier

        self.estimator = GaussianProcessClassifier(
            kernel=self.kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            multi_class=self.multi_class
        )
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GP',
                'name': 'Gaussian Process Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)}
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        kernel = CategoricalHyperparameter("kernel",
                                           ["constant", "rbf", "matern", "rational_quadratic", "exp_sine_squared"],
                                           default_value="rbf")
        optimizer = Constant("optimizer", "fmin_l_bfgs_b")
        n_restarts_optimizer = UniformIntegerHyperparameter("n_restarts_optimizer", 0, 1000, default_value=0)
        max_iter_predict = UniformIntegerHyperparameter("max_iter_predict", 1, 1000, default_value=100)
        warm_start = CategoricalHyperparameter("warm_start", [True, False], default_value=False)
        copy_X_train = CategoricalHyperparameter("copy_X_train", [True, False], default_value=True)
        multi_class = CategoricalHyperparameter("multi_class", ["one_vs_rest", "one_vs_one"],
                                                default_value="one_vs_rest")

        cs.add_hyperparameters(
            [n_restarts_optimizer, max_iter_predict, warm_start, copy_X_train, multi_class,kernel,alpha,optimizer])

        return cs
