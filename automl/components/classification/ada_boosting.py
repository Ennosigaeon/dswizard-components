from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS


class AdaBoostingClassifier(PredictionAlgorithm):
    def __init__(self,
                 algorithm: str = 'SAMME.R',
                 learning_rate: float = 1.0,
                 n_estimators: int = 50,
                 random_state=None,
                 ):
        super().__init__()
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(
            algorithm=self.algorithm,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AB',
                'name': 'Ada Boosting Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=1e-6, upper=15., default_value=1.0,
                                                   log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 1500, default_value=50, log=False)
        algorithm = CategoricalHyperparameter("algorithm", ["SAMME", "SAMME.R"], default_value="SAMME.R")

        cs.add_hyperparameters(
            [learning_rate, n_estimators, algorithm])

        return cs
