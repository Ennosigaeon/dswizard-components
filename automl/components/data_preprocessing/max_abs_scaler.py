from ConfigSpace.configuration_space import ConfigurationSpace

from automl.components.base import PreprocessingAlgorithm


class MaxAbsScalerComponent(PreprocessingAlgorithm):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        from sklearn.preprocessing import MaxAbsScaler
        self.preprocessor = MaxAbsScaler(copy=False)
        self.preprocessor.fit(X)
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MaxAbsScaler',
                'name': 'MaxAbsScaler',
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
