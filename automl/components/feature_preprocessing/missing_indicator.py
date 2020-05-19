import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from automl.components.base import PreprocessingAlgorithm


class MissingIndicatorComponent(PreprocessingAlgorithm):
    def __init__(self, missing_values=np.nan, features: str = "missing-only"):
        super().__init__()

        self.features = features
        self.missing_values = missing_values

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.impute import MissingIndicator
        return MissingIndicator(missing_values=self.missing_values, features=self.features)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                'handles_missing_values': True,
                'handles_nominal_values': True,
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
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        features = CategoricalHyperparameter("features", ["missing-only", "all"], default_value="missing-only")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features])
        return cs
