from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenInClause
from ConfigSpace.conditions import InCondition, AndConjunction
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class LogisticRegression(PredictionAlgorithm):
    def __init__(self,
                 penalty: str = 'l2',
                 solver: str = 'lbfgs',
                 dual: bool = False,
                 tol: float = 1e-4,
                 C: float = 1.0,
                 fit_intercept: bool = True,
                 intercept_scaling: float = 1,
                 max_iter: int = 100,
                 multi_class: str = "auto",
                 l1_ratio: float = None,
                 random_state=None
                 ):
        super().__init__()
        self.penalty = penalty
        self.solver = solver
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1])
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.linear_model import LogisticRegression

        # Cast only for test case
        if self.intercept_scaling == 1.:
            self.intercept_scaling = 1

        return LogisticRegression(
            penalty=self.penalty,
            solver=self.solver,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            random_state=self.random_state,
            l1_ratio=self.l1_ratio)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'LR',
                'name': 'Logistic Regression',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        penalty = CategoricalHyperparameter("penalty", ["l1", "l2", "elasticnet", "none"], default_value='l2')
        solver = CategoricalHyperparameter("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                                           default_value="lbfgs")
        dual = CategoricalHyperparameter("dual", choices=[True, False], default_value=False)
        tol = UniformFloatHyperparameter("tol", lower=1e-7, upper=100., default_value=1.0e-4, log=True)
        C = UniformFloatHyperparameter("C", lower=1e-7, upper=100., default_value=1.0, log=True)
        fit_intercept = CategoricalHyperparameter("fit_intercept", choices=[True, False], default_value=True)
        intercept_scaling = UniformFloatHyperparameter("intercept_scaling", lower=0.0001, upper=2.0, default_value=1.0,
                                                       log=True)
        max_iter = UniformIntegerHyperparameter("max_iter", lower=50, upper=10000, default_value=100)
        multi_class = CategoricalHyperparameter("multi_class", ["ovr", "multinomial", "auto"], default_value="auto")
        l1_ratio = UniformFloatHyperparameter("l1_ratio", lower=0., upper=1., default_value=0.1)

        l1_ratio_condition = InCondition(l1_ratio, penalty, ["elasticnet"])
        dual_condition = AndConjunction(InCondition(dual, penalty, ["l2"]), InCondition(dual, solver, ["liblinear"]))
        cs.add_hyperparameters([penalty, solver, dual, tol, C, fit_intercept, intercept_scaling, max_iter, multi_class,
                                l1_ratio])

        penaltyAndLbfgs = ForbiddenAndConjunction(
            ForbiddenEqualsClause(solver, "lbfgs"),
            ForbiddenInClause(penalty, ["l1", "elasticnet"])
        )
        penaltyAndNewton = ForbiddenAndConjunction(
            ForbiddenEqualsClause(solver, "newton-cg"),
            ForbiddenInClause(penalty, ["l1", "elasticnet"])
        )
        penaltyAndSag = ForbiddenAndConjunction(
            ForbiddenEqualsClause(solver, "sag"),
            ForbiddenInClause(penalty, ["l1", "elasticnet"])
        )
        penaltyAndSaga = ForbiddenAndConjunction(
            ForbiddenInClause(penalty, ["elasticnet"]),
            ForbiddenInClause(solver, ["newton-cg", "lbfgs", "sag"])
        )
        penaltyAndSagaa = ForbiddenAndConjunction(
            ForbiddenInClause(penalty, ["elasticnet", "none"]),
            ForbiddenInClause(solver, ["liblinear"])
        )
        penaltyAndSagaaa = ForbiddenAndConjunction(
            ForbiddenInClause(multi_class, ["multinomial"]),
            ForbiddenInClause(solver, ["liblinear"])
        )

        cs.add_forbidden_clause(penaltyAndLbfgs)
        cs.add_forbidden_clause(penaltyAndNewton)
        cs.add_forbidden_clause(penaltyAndSag)
        cs.add_forbidden_clause(penaltyAndSagaa)
        cs.add_forbidden_clause(penaltyAndSaga)
        cs.add_forbidden_clause(penaltyAndSagaaa)
        cs.add_condition(l1_ratio_condition)
        cs.add_condition(dual_condition)
        return cs
