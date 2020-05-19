from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter
from scipy.sparse import csr_matrix

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class KBinsDiscretizer(PreprocessingAlgorithm):

    def __init__(self, n_bins: int = 5,
                 encode: str = "onehot",
                 strategy: str = "quantile"):
        super().__init__()
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import KBinsDiscretizer
        return KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)

    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError()
        Xt = self.preprocessor.transform(X)

        # TODO sparse matrix currently not supported
        if isinstance(Xt, csr_matrix):
            return Xt.todense()
        else:
            return Xt

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_bins = UniformIntegerHyperparameter("n_bins", 2, 100, default_value=5)
        encode = CategoricalHyperparameter("encode", ["onehot", "onehot-dense", "ordinal"], default_value="onehot")
        strategy = CategoricalHyperparameter("strategy", ["uniform", "quantile", "kmeans"], default_value="quantile")

        cs.add_hyperparameters([n_bins, encode, strategy])
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        # TODO find out of this is right!
        return {'shortname': 'KBD',
                'name': 'K Bins Discretizer',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
