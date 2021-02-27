from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class PassiveAggressiveClassifier(PredictionAlgorithm):

    def __init__(self,
                 C: float = 1.0,
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 early_stopping: bool = False,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 5,
                 shuffle: bool = True,
                 loss: str = "hinge",
                 average: int = False,
                 random_state=None
                 ):
        super().__init__()
        self.C = C
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.loss = loss
        self.average = average
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.linear_model import PassiveAggressiveClassifier

        average = False if self.average == 1 else self.average

        return PassiveAggressiveClassifier(
            C=self.C,
            fit_intercept=self.fit_intercept,
            tol=self.tol,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            shuffle=self.shuffle,
            loss=self.loss,
            average=average,
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
        return {'shortname': 'PA',
                'name': 'Passive Aggressive Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        C = UniformFloatHyperparameter("C", 1e-6, 25., default_value=1.)
        fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
        max_iter = UniformIntegerHyperparameter("max_iter", 5, 1000, default_value=1000)
        tol = UniformFloatHyperparameter("tol", 1e-7, 1., default_value=1e-3)
        early_stopping = CategoricalHyperparameter("early_stopping", [True, False], False)
        validation_fraction = UniformFloatHyperparameter("validation_fraction", 0., 1., default_value=0.1)
        n_iter_no_change = UniformIntegerHyperparameter("n_iter_no_change", 1, 1000, default_value=5)
        shuffle = CategoricalHyperparameter("shuffle", [True, False], True)
        loss = CategoricalHyperparameter("loss", ["hinge", "squared_hinge"], default_value="hinge")
        average = UniformIntegerHyperparameter("average", 1, 100, default_value=1)

        cs.add_hyperparameters([C, fit_intercept, max_iter, tol, early_stopping, validation_fraction, n_iter_no_change,
                                shuffle, loss, average])

        validation_fraction_condition = EqualsCondition(validation_fraction, early_stopping, True)
        cs.add_condition(validation_fraction_condition)

        return cs
