from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class SelectPercentileClassification(PreprocessingAlgorithm):
    """SelectPercentile removes all but a user-specified highest scoring percentage of features. It provides an
        automatic procedure for keeping only a certain percentage of the best, associated features."""

    def __init__(self,
                 percentile: float = 10,
                 score_func: str = "f_classif",
                 random_state=None):
        """ Parameters:
        random state : ignored

        score_func : callable, Function taking two arrays X and y, and
                     returning a pair of arrays (scores, pvalues).
        """

        super().__init__()

        self.random_state = random_state  # We don't use this
        self.percentile = percentile
        self.score_func = score_func

    def fit(self, X, y):
        import scipy.sparse
        from sklearn.feature_selection import chi2

        self.preprocessor = self.to_sklearn(X.shape[0], X.shape[1])
        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        self.preprocessor.fit(X, y)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif

        if self.score_func == "chi2":
            score_func = chi2
        elif self.score_func == "f_classif":
            score_func = f_classif
        elif self.score_func == "mutual_info":
            score_func = mutual_info_classif
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info'), but is: %s" % self.score_func)

        return SelectPercentile(score_func=score_func,
                                percentile=self.percentile)

    def transform(self, X):
        import scipy.sparse
        import sklearn.feature_selection

        # TODO really? I assume only copied from auto-sklearn
        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        if self.preprocessor is None:
            raise NotImplementedError()
        Xt = self.preprocessor.transform(X)
        if Xt.shape[1] == 0:
            raise ValueError(
                "%s removed all features." % self.__class__.__name__)
        return Xt

    @staticmethod
    def get_properties():
        return {'shortname': 'SPC',
                'name': 'Select Percentile Classification',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        percentile = UniformIntegerHyperparameter(name="percentile", lower=1, upper=99, default_value=10)

        score_func = CategoricalHyperparameter(name="score_func", choices=["chi2", "f_classif", "mutual_info"],
                                               default_value="f_classif")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([percentile, score_func])

        return cs
