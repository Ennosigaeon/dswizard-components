from automl.components.base import PreprocessingAlgorithm
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

class QuantileTransformerComponent(PreprocessingAlgorithm):

    def __init__(self, n_quantiles: int = 1000, output_distribution: str = "uniform", ignore_implicit_zeros: bool = False, subsample: int = 1e5, copy: bool = True, random_state=None):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.copy = copy
        self.random_state = random_state
        from sklearn.preprocessing import QuantileTransformer
        self.preprocessor = QuantileTransformer(copy=self.copy,
                                                n_quantiles=self.n_quantiles,
                                                output_distribution=self.output_distribution,
                                                ignore_implicit_zeros=self.ignore_implicit_zeros,
                                                subsample=self.subsample,random_state=self.random_state)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_quantiles = UniformIntegerHyperparameter("n_quantiles", 10, 2500, default_value=1000)
        output_distribution = CategoricalHyperparameter("output_distribution", ["uniform", "normal"],
                                                        default_value="uniform")
        ignore_implicit_zeros = CategoricalHyperparameter("ignore_implicit_zeros", [True, False], default_value=False)
        subsample = UniformIntegerHyperparameter("subsample", 1e3, 1e8, default_value=1e5)
        copy = CategoricalHyperparameter("copy", [True, False], default_value=True)

        cs.add_hyperparameters([n_quantiles, output_distribution, ignore_implicit_zeros, subsample, copy])
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'QuantileTransformer',
                'name': 'QuantileTransformer',
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
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}
