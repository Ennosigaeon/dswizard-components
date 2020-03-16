from scipy import sparse

from automl.components.base import PreprocessingAlgorithm


class FastICAComponent(PreprocessingAlgorithm):
    def __init__(self):
        super().__init__()
        from sklearn.decomposition import FastICA
        self.preprocessor = FastICA()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FastICA',
                'name': 'FastICA',
                # TODO Check if True
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
