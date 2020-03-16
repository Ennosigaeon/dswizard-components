import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class SVCClassifier(PredictionAlgorithm):

    def __init__(self, C: float = 1.0,
                 kernel: str = "rbf",
                 degree: int = 3,
                 gamma: float = 0.5,
                 coef0: float = 0.,
                 shrinking: bool = True,
                 probability: bool = False,
                 tol: float = 1e-3,
                 max_iter: int = -1,
                 decision_function_shape: str = "ovr",
                 break_ties: bool = False,
                 ):
        super().__init__()
        self.C = C,
        self.kernel = kernel,
        self.degree = degree,
        self.gamma = gamma,
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties

    def fit(self, X, y):
        from sklearn.svm import SVC

        self.estimator = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape
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
        return {'shortname': 'SVC',
                'name': 'Support Vector Classifier',
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

        C = UniformFloatHyperparameter("C", 0., 10., default_value=1.)
        kernel = CategoricalHyperparameter("kernel", ["linear", "poly", "rbf", "sigmoid"], default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", 2, 10, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 0., 1., default_value=0.5)
        coef0 = UniformFloatHyperparameter("coef0", 0., 5., default_value=0.)
        shrinking = CategoricalHyperparameter("shrinking", [True, False], default_value=True)
        probability = CategoricalHyperparameter("probability", [True, False], default_value=False)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1., default_value=1e-3)
        max_iter = UniformIntegerHyperparameter("max_iter", -1, 1000, default_value=-1)
        decision_function_shape = CategoricalHyperparameter("decision_function_shape", ["ovr", "ovo"],
                                                            default_value="ovr")
        break_ties = CategoricalHyperparameter("break_ties", [True, False], default_value=False)

        cs.add_hyperparameters(
            [C, kernel, degree, gamma, coef0, shrinking, probability, tol, max_iter, decision_function_shape,
             break_ties])

        return cs
