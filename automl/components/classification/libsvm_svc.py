import resource
import sys

from ConfigSpace import ForbiddenAndConjunction, ForbiddenInClause
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none, check_for_bool
from automl.util.util import softmax


# noinspection PyPep8Naming
class LibSVM_SVC(PredictionAlgorithm):
    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: str = 'scale',
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 tol: float = 1e-3,
                 class_weight=None,
                 random_state=None,
                 decision_function_shape: str = "ovr",
                 break_ties: bool = False,
                 probability: bool = False
                 ):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.decision_function_shape = decision_function_shape
        self.probability = probability
        self.break_ties = break_ties

    def fit(self, X, Y):
        from sklearn.svm import SVC

        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM. The cache size is
        # calculated as 2/3 of the available memory (which is calculated as the memory limit minus
        # the used memory)
        try:
            # Retrieve memory limits imposed on the process
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)

            if soft > 0:
                # Convert limit to units of megabytes
                soft /= 1024 * 1024

                # Retrieve memory used by this process
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024

                # In MacOS, the MaxRSS output of resource.getrusage in bytes; on other platforms,
                # it's in kilobytes
                if sys.platform == 'darwin':
                    maxrss = maxrss / 1024

                cache_size = (soft - maxrss) / 1.5

                if cache_size < 0:
                    cache_size = 200
            else:
                cache_size = 200
        except Exception:
            cache_size = 200

        if self.degree is None:
            self.degree = 3
        if self.gamma is None:
            self.gamma = 0.0
        if self.coef0 is None:
            self.coef0 = 0.0

        self.shrinking = check_for_bool(self.shrinking)

        if check_none(self.class_weight):
            self.class_weight = None

        self.estimator = SVC(C=self.C,
                             kernel=self.kernel,
                             degree=self.degree,
                             gamma=self.gamma,
                             coef0=self.coef0,
                             shrinking=self.shrinking,
                             tol=self.tol,
                             class_weight=self.class_weight,
                             random_state=self.random_state,
                             cache_size=cache_size,
                             probability=self.probability,
                             break_ties=self.break_ties,
                             decision_function_shape=self.decision_function_shape
                             )
        self.estimator.fit(X, Y)
        return self

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LibSVM-SVC',
                'name': 'LibSVM Support Vector Classification',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)}
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 1e-7, 2500, default_value=1.0, log=True)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(name="kernel", choices=["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", 2, 6, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 1e-7, 1, log=True, default_value=0.1)
        coef0 = UniformFloatHyperparameter("coef0", -30., 30., default_value=0.)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter("shrinking", [True, False], default_value=True)
        probability = CategoricalHyperparameter("probability", [True, False], default_value=False)
        tol = UniformFloatHyperparameter("tol", 1e-7, 0.5, default_value=1e-3, log=True)
        # cache size is not a hyperparameter, but an argument to the program!
        decision_function_shape = CategoricalHyperparameter("decision_function_shape", ["ovr", "ovo"],
                                                            default_value="ovr")
        break_ties = CategoricalHyperparameter("break_ties", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking, tol, probability,
                                decision_function_shape, break_ties])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["rbf", "poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)
        cs.add_condition(gamma_condition)

        funcshapAndBreakties = ForbiddenAndConjunction(
            ForbiddenInClause(decision_function_shape, ["ovo"]),
            ForbiddenInClause(break_ties, [True])
        )
        cs.add_forbidden_clause(funcshapAndBreakties)

        return cs
