from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from automl.components.base import PreprocessingAlgorithm


class NormalizerComponent(PreprocessingAlgorithm):
    def __init__(self, norm: str = 'l2'):
        super().__init__()
        self.norm = norm

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.preprocessing import Normalizer
        return Normalizer(norm=self.norm, copy=False)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        norm = CategoricalHyperparameter('norm', ['l1', 'l2', 'max'], default_value='l2')
        cs.add_hyperparameter(norm)
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Normalizer',
                'name': 'Normalizer',
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
                # 'input': (SPARSE, DENSE, UNSIGNED_DATA),
                # 'output': (INPUT,),
                'preferred_dtype': None}
