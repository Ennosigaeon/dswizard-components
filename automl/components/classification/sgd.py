import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition, InCondition

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class SGDClassifier(PredictionAlgorithm):

    def __init__(self,
                 loss: str = "hinge",
                 penalty: str = "l2",
                 alpha: float = 0.001,
                 l1_ratio: float = 0.15,
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 shuffle: bool = True,
                 epsilon: float = 0.1,
                 learning_rate: str = "optimal",
                 eta0: float = 0.,
                 power_t: float = 0.5,
                 early_stopping: bool = False,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 5,
                 warm_start: bool = False,
                 average: bool = False,
                 random_state = None
                 ):
        super().__init__()
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.linear_model import SGDClassifier

        self.estimator = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            epsilon=self.epsilon,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            warm_start=self.warm_start,
            average=self.average,
            random_state=self.random_state
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
        return {'shortname': 'SGD',
                'name': 'Stochastic Gradient Descent Classifier',
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

        loss = CategoricalHyperparameter("loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron",
                                                  "squared_loss", "huber", "epsilon_insensitive",
                                                  "squared_epsilon_insensitive"], default_value="hinge")
        penaly = CategoricalHyperparameter("penalty", ["l2", "l1", "elasticnet"], default_value="l2")
        alpha = UniformFloatHyperparameter("alpha", 1e-7, 50., default_value=0.001)
        l1_ratio = UniformFloatHyperparameter("l1_ratio", 1e-7, 1., default_value=0.15)
        fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
        max_iter = UniformIntegerHyperparameter("max_iter", 20, 1000, default_value=1000)
        tol = UniformFloatHyperparameter("tol", 1e-9, 1., default_value=1e-3)
        shuffle = CategoricalHyperparameter("shuffle", [True, False], default_value=True)
        epsilon = UniformFloatHyperparameter("epsilon", 1e-9, 50., default_value=0.1)
        learning_rate = CategoricalHyperparameter("learning_rate", ["constant", "optimal", "invscaling", "adaptive"],
                                                  default_value="optimal")
        eta0 = UniformFloatHyperparameter("eta0", 1e-9, 50., default_value=1e-9)
        power_t = UniformFloatHyperparameter("power_t", 1e-9, 50., default_value=0.5)
        early_stopping = CategoricalHyperparameter("early_stopping", [True, False], default_value=False)
        validation_fraction = UniformFloatHyperparameter("validation_fraction", 0., 1., default_value=0.1)
        n_iter_no_change = UniformIntegerHyperparameter("n_iter_no_change", 1, 100, default_value=5)
        warm_start = CategoricalHyperparameter("warm_start", [True, False], default_value=False)
        average = CategoricalHyperparameter("average", [True, False], default_value=False)

        cs.add_hyperparameters(
            [loss, penaly, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, epsilon, learning_rate, eta0,
             power_t, early_stopping, validation_fraction, n_iter_no_change, warm_start, average])

        eta0_condition = InCondition(eta0, learning_rate, ["constant", "invscaling", "adaptive"])
        validation_fraction_condition = EqualsCondition(validation_fraction, early_stopping, True)
        cs.add_condition(eta0_condition)
        cs.add_condition(validation_fraction_condition)

        return cs
