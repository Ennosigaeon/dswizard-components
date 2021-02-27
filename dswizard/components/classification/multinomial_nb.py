from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import check_for_bool, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, \
    HANDLES_MISSING, HANDLES_NOMINAL_CLASS


class MultinomialNB(PredictionAlgorithm):

    def __init__(self,
                 alpha: float = 1.0,
                 fit_prior: bool = True,
                 verbose: int = 0):
        super().__init__()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.verbose = int(verbose)

    def fit(self, X, y):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1], len(y.shape) > 1 and y.shape[1] > 1)
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, multilabel: bool = False, **kwargs):
        from sklearn.naive_bayes import MultinomialNB
        self.fit_prior = check_for_bool(self.fit_prior)
        estimator = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)

        # Fallback for multilabel classification
        if multilabel:
            import sklearn.multiclass
            estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        return estimator

    @staticmethod
    def get_properties():
        return {'shortname': 'MultinomialNB',
                'name': 'Multinomial Naive Bayes classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: False
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        # the smoothing parameter is a non-negative float
        # I will limit it to 1000 and put it on a logarithmic scale. (SF)
        # Please adjust that, if you know a proper range, this is just a guess.
        alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100., default_value=1, log=True)
        fit_prior = CategoricalHyperparameter(name="fit_prior", choices=[True, False], default_value=True)

        cs.add_hyperparameters([alpha, fit_prior])
        return cs
