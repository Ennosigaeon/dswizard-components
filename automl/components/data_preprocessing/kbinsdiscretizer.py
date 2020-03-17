from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from automl.components.base import PreprocessingAlgorithm


class BinarizerComponent(PreprocessingAlgorithm):

    def __init__(self, n_bins: int = 5,
                 encode: str = "onehot",
                 strategy: str = "quantile"):
        super().__init__()
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def fit(self, X, y=None):
        from sklearn.preprocessing import KBinsDiscretizer
        self.preprocessor = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)
        self.preprocessor = self.preprocessor.fit(X)
        return self

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
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}
