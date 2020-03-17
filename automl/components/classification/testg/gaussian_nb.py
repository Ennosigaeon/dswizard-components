import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from automl.components.base import PredictionAlgorithm


class GaussianNB(PredictionAlgorithm):

    def __init__(self, random_state=None, var_smooting: float = 1e-9, verbose: int = 0):
        super().__init__()
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None
        self.classes_ = None
        self.var_smooting = var_smooting


    def fit(self, X, y):
        import sklearn.naive_bayes

        self.estimator = sklearn.naive_bayes.GaussianNB(var_smoothing=self.var_smooting)
        self.classes_ = np.unique(y.astype(int))

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator = sklearn.multiclass.OneVsRestClassifier(self.estimator, n_jobs=1)
        self.estimator.fit(X, y)

        return self

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

        var_smooting = UniformFloatHyperparameter("var_smoothing", 1e-11, 1., default_value=1e-9)

        return cs
