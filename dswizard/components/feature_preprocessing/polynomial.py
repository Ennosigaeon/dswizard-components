from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import check_for_bool, HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, \
    HANDLES_NUMERIC, HANDLES_MULTICLASS


class PolynomialFeaturesComponent(PreprocessingAlgorithm):
    def __init__(self, degree: int = 2, interaction_only: bool = False, include_bias: bool = True, order: str = "C"):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import PolynomialFeatures

        self.interaction_only = check_for_bool(self.interaction_only)
        self.include_bias = check_for_bool(self.include_bias)

        return PolynomialFeatures(degree=self.degree,
                                  interaction_only=self.interaction_only,
                                  include_bias=self.include_bias,
                                  order=self.order)

    @staticmethod
    def get_properties():
        return {'shortname': 'PolynomialFeatures',
                'name': 'PolynomialFeatures',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        # More than degree 3 is too expensive!
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter("interaction_only", [False, True], False)
        include_bias = CategoricalHyperparameter("include_bias", [True, False], True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs
