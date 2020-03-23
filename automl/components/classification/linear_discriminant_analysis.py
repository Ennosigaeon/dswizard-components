from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from ConfigSpace.conditions import InCondition
from automl.components.base import PredictionAlgorithm
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class LinearDiscriminantAnalysis(PredictionAlgorithm):
    def __init__(self,
                 solver: str = 'svd',
                 shrinkage: str = None,
                 store_covariance: bool = False,
                 tol: float = 1.0e-4,
                 n_components: int = None
                 ):
        super().__init__()
        self.solver = solver
        self.shrinkage = shrinkage
        self.store_covariance = store_covariance
        self.tol = tol
        self.n_components = n_components

    def fit(self, X, y, sample_weight=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # initial fit of only increment trees
        self.estimator = LinearDiscriminantAnalysis(solver=self.solver,
                                                    shrinkage=self.shrinkage,
                                                    store_covariance=self.store_covariance,
                                                    tol=self.tol,
                                                    n_components=self.n_components)
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
        return {'shortname': 'LDA',
                'name': 'Linear Discriminant Analysis',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        shrinkage = UniformFloatHyperparameter("shrinkage", 0., 1., default_value=0.1)  # oder float zw 1 & 0
        solver = CategoricalHyperparameter("solver", ["svd", "lsqr", "eigen"], default_value="svd")
        n_components = UniformIntegerHyperparameter("n_components", 2, 400, default_value=10)
        store_covariance = CategoricalHyperparameter("store_covariance", choices=[True, False], default_value=False)
        tol = UniformFloatHyperparameter(name="tol", lower=1.0e-7, upper=1., default_value=1.0e-4, log=True)

        cs.add_hyperparameters([shrinkage, solver, store_covariance, tol, n_components])

        shrinkage_condition = InCondition(shrinkage, solver, ["lsqr", "eigen"])
        cs.add_condition(shrinkage_condition)

        return cs
