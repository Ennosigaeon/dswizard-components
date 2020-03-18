from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter, Constant

from automl.components.base import PreprocessingAlgorithm


class GenericUnivariateSelectComponent(PreprocessingAlgorithm):

    """Feature selector that removes all low-variance features.
    Features with a training-set variance lower than this threshold will be removed. The default is to keep all
    features with non-zero variance, i.e. remove the features that have the same value in all samples."""

    def __init__(self, alpha: float = 0.5,
                 score_func: str = "chi2",
                 mode: str = "percentile"):
        super().__init__()
        self.alpha = alpha
        self.mode = mode

        import sklearn.feature_selection

        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "f_regression":
            self.score_func = sklearn.feature_selection.f_regression
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info'), "
                             "but is: %s" % score_func)

    def fit(self, X, y=None):
        from sklearn.feature_selection import GenericUnivariateSelect

        self.preprocessor = GenericUnivariateSelect(param=self.alpha,
                                                    mode=self.mode,
                                                    score_func=self.score_func)

        self.preprocessor = self.preprocessor.fit(X, y)
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        mode = CategoricalHyperparameter("mode", ['percentile', 'k_best', 'fpr', 'fdr', 'fwe'], default_value="percentile")
        alpha = UniformFloatHyperparameter("alpha", 0.0001, 0.75, default_value=0.5)
        score_func = CategoricalHyperparameter(
            name="score_func",
            choices=["chi2", "f_classif", "f_regression"],
            default_value="chi2")
        if dataset_properties is not None:
            # Chi2 can handle sparse data, so we respect this
            if 'sparse' in dataset_properties and dataset_properties['sparse']:
                score_func = Constant(
                    name="score_func", value="chi2")

        cs.add_hyperparameters([mode, alpha, score_func])
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'GenericUnivariateSelect',
            'name': 'Generic Univariate Select',
            # TODO Check if True
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'is_deterministic': True,
            'handles_sparse': True,
            'handles_dense': True,
            # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
            # 'output': (INPUT,),
        }

