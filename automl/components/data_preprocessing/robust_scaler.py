from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm


class RobustScalerComponent(PreprocessingAlgorithm):
    def __init__(self, q_min: float = 25.0, q_max: float = 75.0, with_centering: bool = True, with_scaling: bool = True):
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.q_min = q_min
        self.q_max = q_max

        if self.q_max < self.q_min:
            help = self.q_max
            self.q_max = self.q_min
            self.q_min = help

    def fit(self, X, y=None):
        from sklearn.preprocessing import RobustScaler
        self.preprocessor = RobustScaler(quantile_range=(self.q_min, self.q_max), with_centering=self.with_centering, with_scaling=self.with_scaling)
        return super().fit(X, y)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RobustScaler',
                'name': 'RobustScaler',
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

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        with_centering = CategoricalHyperparameter("with_centering", [True,False], default_value=True)
        with_scaling = CategoricalHyperparameter("with_scaling", [True,False], default_value=True)
        q_min = UniformFloatHyperparameter('q_min', 0., 100., default_value=25.)
        q_max = UniformFloatHyperparameter('q_max', 0., 100., default_value=75.)
        cs.add_hyperparameters([q_min, q_max, with_centering,with_scaling,copy])
        return cs
