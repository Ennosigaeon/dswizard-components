import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition, InCondition, LessThanCondition, AndConjunction

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class MLPClassifier(PredictionAlgorithm):

    def __init__(self,
                 activation: str = "relu",
                 solver: str = "relu",
                 alpha: float = 0.0001,
                 batch_size: int = 200,
                 learning_rate: str = "constant",
                 learning_rate_init: float = 0.001,
                 power_t: float = 0.5,
                 max_iter: int = 200,
                 shuffle: bool = True,
                 tol: float = 1e-4,
                 warm_start: bool = True,
                 momentum: float = 0.9,
                 nesterovs_momentum: bool = True,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 beta_1: float = 0.1,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 n_iter_no_change: int = 10,
                 max_fun: int = 15000,
                 layer_1_size: int = 3,
                 layer_2_size: int = 150
                 ):
        super().__init__()
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size
        self.activation = activation
        self.alpha = alpha
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    def fit(self, X, y):
        from sklearn.neural_network import MLPClassifier

        self.estimator = MLPClassifier(
            hidden_layer_sizes=(self.layer_1_size,self.layer_2_size),
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            tol=self.tol,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change,
            max_fun=self.max_fun
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
        return {'shortname': 'MLP',
                'name': 'MLP Classifier',
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

        layer_1_size = UniformIntegerHyperparameter("layer_1_size", 1, 5, default_value=3)
        layer_2_size = UniformIntegerHyperparameter("layer_2_size", 1, 500, default_value=150)
        activation = CategoricalHyperparameter("activation", ["identity", "logistic", "tanh", "relu"],
                                               default_value="relu")
        solver = CategoricalHyperparameter("solver", ["lbfgs", "sgd", "adam"], default_value="adam")
        alpha = UniformFloatHyperparameter("alpha", 0.00001, 2., default_value=0.0001)
        batch_size = UniformIntegerHyperparameter("batch_size", 5, 200, default_value=200)
        learning_rate = CategoricalHyperparameter("learning_rate", ["constant", "invscaling", "adaptive"],
                                                  default_value="constant")
        learning_rate_init = UniformFloatHyperparameter("learning_rate_init", 0.0001, 1., default_value=0.001)
        power_t = UniformFloatHyperparameter("power_t", 0.01, 0.99, default_value=0.5)
        max_iter = UniformIntegerHyperparameter("max_iter", 5, 1000, default_value=200)
        shuffle = CategoricalHyperparameter("shuffle", [True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", 1e-6, 1., default_value=1e-4)
        warm_start = CategoricalHyperparameter("warm_start", [True, False], default_value=True)
        momentum = UniformFloatHyperparameter("momentum", 0., 1., default_value=0.9)
        nesterovs_momentum = CategoricalHyperparameter("nesterovs_momentum", [True, False], default_value=True)
        early_stopping = CategoricalHyperparameter("early_stopping", [True, False], default_value=True)
        validation_fraction = UniformFloatHyperparameter("validation_fraction", 0., 1., default_value=0.1)
        beta_1 = UniformFloatHyperparameter("beta_1", 0., 0.9999, default_value=0.1)
        beta_2 = UniformFloatHyperparameter("beta_2", 0., 0.9999, default_value=0.999)
        epsilon = UniformFloatHyperparameter("epsilon", 0., 1., default_value=1e-8)
        n_iter_no_change = UniformIntegerHyperparameter("n_iter_no_change", 1, 1000, default_value=10)
        max_fun = UniformIntegerHyperparameter("max_fun", 200, 100000, default_value=15000)

        cs.add_hyperparameters(
            [layer_1_size, layer_2_size, activation, solver, alpha, batch_size, learning_rate, learning_rate_init,
             power_t, max_iter, shuffle, tol, warm_start, momentum, nesterovs_momentum, early_stopping,
             validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun])

        learning_rate_init_condition = InCondition(learning_rate_init, solver, ["sgd", "adam"])
        power_t_condition = InCondition(power_t, solver, ["sgd"])
        power_t_2_condition = InCondition(power_t, learning_rate, ["invscaling"])
        momentum_condition = InCondition(momentum, solver, ["sgd"])
        nesterovs_momentum_condition = LessThanCondition(nesterovs_momentum, momentum, 0.)
        nesterovs_momentum_condition_2 = InCondition(nesterovs_momentum, solver, ["sgd"])
        early_stopping_condition = InCondition(early_stopping, solver, ["sgd", "adam"])
        validation_fraction = EqualsCondition(validation_fraction, early_stopping, True)
        beta_1_condition = InCondition(beta_1, solver, ["adam"])
        beta_2_condition = InCondition(beta_2, solver, ["adam"])
        epsilon_condition = InCondition(epsilon, solver, ["adam"])
        n_iter_no_change_condition = InCondition(n_iter_no_change, solver, ["sgd", "adam"])

        cs.add_condition(learning_rate_init_condition)
        cs.add_condition(AndConjunction(power_t_condition, power_t_2_condition))
        cs.add_condition(momentum_condition)
        cs.add_condition(AndConjunction(nesterovs_momentum_condition, nesterovs_momentum_condition_2))
        cs.add_condition(early_stopping_condition)
        cs.add_condition(validation_fraction)
        cs.add_condition(beta_1_condition)
        cs.add_condition(beta_2_condition)
        cs.add_condition(epsilon_condition)
        cs.add_condition(n_iter_no_change_condition)

        return cs
