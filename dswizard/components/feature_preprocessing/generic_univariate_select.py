from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class GenericUnivariateSelectComponent(PreprocessingAlgorithm):
    """Feature selector that removes all low-variance features.
    Features with a training-set variance lower than this threshold will be removed. The default is to keep all
    features with non-zero variance, i.e. remove the features that have the same value in all samples."""

    def __init__(self, param: float = 1e-05,
                 score_func: str = "f_classif",
                 mode: str = "percentile"):
        super().__init__()
        self.param = param
        self.mode = mode
        self.score_func = score_func

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.feature_selection import GenericUnivariateSelect

        import sklearn.feature_selection

        if isinstance(self.score_func, str):
            if self.score_func == "chi2":
                score_func = sklearn.feature_selection.chi2
            elif self.score_func == "f_classif":
                score_func = sklearn.feature_selection.f_classif
            elif self.score_func == "f_regression":
                score_func = sklearn.feature_selection.f_regression
            else:
                raise ValueError(
                    "score_func must be in ('chi2, 'f_classif', 'mutual_info'), but is: %s" % self.score_func)
        else:
            score_func = self.score_func

        return GenericUnivariateSelect(param=self.param,
                                       mode=self.mode,
                                       score_func=score_func)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        mode = CategoricalHyperparameter("mode", ['percentile', 'k_best', 'fpr', 'fdr', 'fwe'],
                                         default_value="percentile")
        param = UniformFloatHyperparameter("param", 1e-05, 0.75, default_value=1e-05)
        score_func = CategoricalHyperparameter(name="score_func", choices=["chi2", "f_classif", "f_regression"],
                                               default_value="f_classif")

        cs.add_hyperparameters([mode, param, score_func])
        return cs

    @staticmethod
    def get_properties():
        return {
            'shortname': 'GenericUnivariateSelect',
            'name': 'Generic Univariate Select',
            HANDLES_MULTICLASS: True,
            HANDLES_NUMERIC: True,
            HANDLES_NOMINAL: False,
            HANDLES_MISSING: False,
            HANDLES_NOMINAL_CLASS: True}
