from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import resolve_factor, HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class QuantileTransformerComponent(PreprocessingAlgorithm):

    def __init__(self, n_quantiles_factor: int = None, output_distribution: str = "uniform",
                 ignore_implicit_zeros: bool = False, subsample: int = int(1e5), random_state=None):
        super().__init__()
        self.n_quantiles_factor = n_quantiles_factor
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import QuantileTransformer

        n_quantiles = resolve_factor(self.n_quantiles_factor, n_samples, default=1000)
        return QuantileTransformer(copy=False,
                                   n_quantiles=n_quantiles,
                                   output_distribution=self.output_distribution,
                                   ignore_implicit_zeros=self.ignore_implicit_zeros,
                                   subsample=self.subsample, random_state=self.random_state)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_quantiles = UniformFloatHyperparameter("n_quantiles_factor", 0., 1., default_value=0.5)
        output_distribution = CategoricalHyperparameter("output_distribution", ["uniform", "normal"],
                                                        default_value="uniform")
        ignore_implicit_zeros = CategoricalHyperparameter("ignore_implicit_zeros", [True, False], default_value=False)
        subsample = UniformIntegerHyperparameter("subsample", 1e3, 1e8, default_value=1e5)

        cs.add_hyperparameters([n_quantiles, output_distribution, ignore_implicit_zeros, subsample])
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'QuantileTransformer',
                'name': 'QuantileTransformer',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
