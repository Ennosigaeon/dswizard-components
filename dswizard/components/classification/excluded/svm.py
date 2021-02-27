from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class SVCClassifier(PredictionAlgorithm):

    def __init__(self,
                 penalty: str = "l2",
                 loss: str = "squared_hinge",
                 dual: bool = True,
                 tol: float = 1e-4,
                 multi_class: str = "ovr",
                 C: float = 1.,
                 fit_intercept: bool = True,
                 intercept_scaling: float = 1,
                 max_iter: int = 1000,
                 random_state=None
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
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.svm import LinearSVC

        # Conversion for test case
        if self.intercept_scaling == 1.0:
            self.intercept_scaling = 1

        return LinearSVC(
            C=self.C,
            penalty=self.penalty,
            loss=self.loss,
            dual=self.dual,
            tol=self.tol,
            multi_class=self.multi_class,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            max_iter=self.max_iter,
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
        return {'shortname': 'LinearSVC',
                'name': 'Linear Support Vector Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        penalty = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l2")
        loss = CategoricalHyperparameter("loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
        dual = CategoricalHyperparameter("dual", [True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", 1e-5, 120., default_value=1e-4)
        C = UniformFloatHyperparameter("C", 1e-7, 100., default_value=1.)
        multi_class = CategoricalHyperparameter("multi_class", ["ovr", "crammer_singer"], default_value="ovr")
        fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
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
