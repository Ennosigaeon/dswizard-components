import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant
from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class SVCClassifier(PredictionAlgorithm):

    def __init__(self,
                 penalty: str = "l2",
                 loss: str = "squared_hinge",
                 dual: bool = True,
                 tol: float = 1e-4,
                 multi_class: str = "ovr",
                 C: float = 1.,
                 fit_intercept: bool = True,
                 intercept_scaling: float = 1.,
                 max_iter: int = 1000
                 ):
        super().__init__()
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.multi_class = multi_class
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.max_iter = max_iter


    def fit(self, X, y):
        from sklearn.svm import LinearSVC

        self.estimator = LinearSVC(
            C=self.C,
            penalty=self.penalty,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            multi_class=self.multi_class,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            max_iter=self.max_iter
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
        return {'shortname': 'LinearSVC',
                'name': 'Linear Support Vector Classifier',
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

        penalty = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l2")
        loss = CategoricalHyperparameter("loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
        dual = CategoricalHyperparameter("dual", [True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", 1e-5, 120., default_value=1e-4)
        C = UniformFloatHyperparameter("C", 1e-7, 100., default_value=1.)
        multi_class = CategoricalHyperparameter("multi_class", ["ovr", "crammer_singer"], default_value="ovr")
        fit_intercept = CategoricalHyperparameter("fit_intercept", [True,False], default_value=True)
        intercept_scaling = UniformFloatHyperparameter("intercept_scaling", 0., 1., default_value=1.)
        max_iter = UniformIntegerHyperparameter("max_iter", 100, 2000, default_value=1000)

        cs.add_hyperparameters(
            [C, penalty, loss, dual, tol, multi_class, fit_intercept, intercept_scaling, max_iter])

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        constant_penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, False),
            ForbiddenEqualsClause(penalty, "l2"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        penalty_and_dual = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, False),
            ForbiddenEqualsClause(penalty, "l1")
        )
        constant_penalty_and_loss2 = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, True),
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "squared_hinge")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        cs.add_forbidden_clause(penalty_and_dual)
        cs.add_forbidden_clause(constant_penalty_and_loss2)

        return cs
