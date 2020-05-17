import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_for_bool


class MultinomialNB(PredictionAlgorithm):

    def __init__(self,
                 alpha: float = 1.0,
                 fit_prior: bool = True,
                 verbose: int = 0):
        super().__init__()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.verbose = int(verbose)
        self.classes_ = None

    def fit(self, X, y):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1], len(y.shape) > 1 and y.shape[1] > 1)
        self.classes_ = np.unique(y.astype(int))
        self.estimator.fit(X, y)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, multilabel: bool = False):
        from sklearn.naive_bayes import MultinomialNB
        self.fit_prior = check_for_bool(self.fit_prior)
        estimator = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)

        # Fallback for multilabel classification
        if multilabel:
            import sklearn.multiclass
            estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MultinomialNB',
                'name': 'Multinomial Naive Bayes classifier',
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

        # the smoothing parameter is a non-negative float
        # I will limit it to 1000 and put it on a logarithmic scale. (SF)
        # Please adjust that, if you know a proper range, this is just a guess.
        alpha = UniformFloatHyperparameter(name="alpha", lower=1e-5, upper=120., default_value=1, log=True)
        fit_prior = CategoricalHyperparameter(name="fit_prior", choices=[True, False], default_value=True)

        cs.add_hyperparameters([alpha, fit_prior])
        return cs
