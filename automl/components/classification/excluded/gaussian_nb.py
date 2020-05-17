import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from automl.components.base import PredictionAlgorithm


class GaussianNB(PredictionAlgorithm):

    def __init__(self, random_state=None,
                 var_smoothing: float = 1e-9,
                 verbose: int = 0):
        super().__init__()
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None
        self.classes_ = None
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1], len(y.shape) > 1 and y.shape[1] > 1)
        self.classes_ = np.unique(y.astype(int))
        self.estimator.fit(X, y)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, multilabel: bool = False):
        import sklearn.naive_bayes

        estimator = sklearn.naive_bayes.GaussianNB(var_smoothing=self.var_smoothing)

        # Fallback for multilabel classification
        if multilabel:
            import sklearn.multiclass
            estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GaussianNB',
                'name': 'Gaussian Naive Bayes classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        var_smoothing = UniformFloatHyperparameter("var_smoothing", 0., 0.25, default_value=1e-9)

        cs.add_hyperparameter(var_smoothing)

        return cs
