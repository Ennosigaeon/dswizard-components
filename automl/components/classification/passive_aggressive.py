from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class PassiveAggressiveClassifier(PredictionAlgorithm):

    def __init__(self,
                 C: float = 1.0,
                 fit_intercept: bool = False,
                 max_iter: int = 1000,
                 tol: float = 1e-3,
                 early_stopping: bool = False,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 5,
                 shuffle: bool = True,
                 loss: str = "hinge",
                 average: int = 1,
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

    def fit(self, X, y):
        from sklearn.linear_model import PassiveAggressiveClassifier

        self.estimator = PassiveAggressiveClassifier(
            C=self.C,
            fit_intercept=self.fit_intercept,
            tol=self.tol,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            shuffle=self.shuffle,
            loss=self.loss,
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
        return {'shortname': 'PA',
                'name': 'Passive Aggressive Classifier',
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

        C = UniformFloatHyperparameter("C", 1e-6, 25., default_value=1.)
        fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=False)
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
