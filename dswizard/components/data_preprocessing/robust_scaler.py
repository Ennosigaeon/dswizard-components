from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS


class RobustScalerComponent(PreprocessingAlgorithm):
    def __init__(self, q_min: float = 25.0, q_max: float = 75.0, with_centering: bool = True,
                 with_scaling: bool = True):
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.q_min = q_min
        self.q_max = q_max

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        if self.q_max < self.q_min:
            help = self.q_max
            self.q_max = self.q_min
            self.q_min = help

        from sklearn.preprocessing import RobustScaler
        return RobustScaler(quantile_range=(self.q_min, self.q_max), copy=False,
                            with_centering=self.with_centering, with_scaling=self.with_scaling)

    @staticmethod
    def get_properties():
        return {'shortname': 'RobustScaler',
                'name': 'RobustScaler',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        q_min = UniformFloatHyperparameter('q_min', 0.1, 30, default_value=25)
        q_max = UniformFloatHyperparameter('q_max', 70, 99.9, default_value=75)
        cs.add_hyperparameters([q_min, q_max])
        return cs
