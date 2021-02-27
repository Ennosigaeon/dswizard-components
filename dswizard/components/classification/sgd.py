from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NOMINAL, HANDLES_NUMERIC, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


# TODO does not honour affinity restrictions

class SGDClassifier(PredictionAlgorithm):

    def __init__(self,
                 loss: str = "hinge",
                 penalty: str = "l2",
                 alpha: float = 0.0001,
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
                 average: bool = False,
                 random_state=None
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
        self.average = average
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier(
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
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            average=self.average,
            n_jobs=1,
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
        return {'shortname': 'SGD',
                'name': 'Stochastic Gradient Descent Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        loss = CategoricalHyperparameter("loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                                         default_value="hinge")
        penalty = CategoricalHyperparameter("penalty", ["l2", "l1", "elasticnet"], default_value="l2")
        alpha = UniformFloatHyperparameter("alpha", 1e-7, 1e-1, default_value=0.0001, log=True)
        l1_ratio = UniformFloatHyperparameter("l1_ratio", 1e-9, 1., default_value=0.15, log=True)
        fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        epsilon = UniformFloatHyperparameter("epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
        learning_rate = CategoricalHyperparameter("learning_rate", ["constant", "optimal", "invscaling"],
                                                  default_value="invscaling")
        eta0 = UniformFloatHyperparameter("eta0", 1e-7, 1e-1, default_value=0.01, log=True)
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, default_value=0.5)
        average = CategoricalHyperparameter("average", [True, False], default_value=False)

        cs.add_hyperparameters(
            [loss, penalty, alpha, l1_ratio, fit_intercept, tol, epsilon, learning_rate, eta0, power_t, average])

        elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
        epsilon_condition = EqualsCondition(epsilon, loss, "modified_huber")

        power_t_condition = EqualsCondition(power_t, learning_rate,
                                            "invscaling")

        # eta0 is only relevant if learning_rate!='optimal' according to code
        # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
        # linear_model/sgd_fast.pyx#L603
        eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling",
                                                            "constant"])
        cs.add_conditions([elasticnet, epsilon_condition, power_t_condition,
                           eta0_in_inv_con])
        return cs
