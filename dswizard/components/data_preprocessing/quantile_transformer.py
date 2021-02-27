from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class QuantileTransformerComponent(PreprocessingAlgorithm):

    def __init__(self, n_quantiles: int = 1000, output_distribution: str = "uniform",
                 ignore_implicit_zeros: bool = False, subsample: int = int(1e5), random_state=None):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import QuantileTransformer

        return QuantileTransformer(copy=False,
                                   n_quantiles=self.n_quantiles,
                                   output_distribution=self.output_distribution,
                                   ignore_implicit_zeros=self.ignore_implicit_zeros,
                                   subsample=self.subsample, random_state=self.random_state)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        n_quantiles = UniformIntegerHyperparameter('n_quantiles', lower=10, upper=2000, default_value=1000)
        output_distribution = CategoricalHyperparameter("output_distribution", ["uniform", "normal"],
                                                        default_value="uniform")

        cs.add_hyperparameters([n_quantiles, output_distribution])
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'QuantileTransformer',
                'name': 'QuantileTransformer',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}
