from scipy import sparse

from automl.components.base import PreprocessingAlgorithm


class TruncatedSVDComponent(PreprocessingAlgorithm):
    def __init__(self):
        super().__init__()
        from sklearn.decomposition import TruncatedSVD
        self.preprocessor = TruncatedSVD()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'TruncatedSVD',
                'name': 'TruncatedSVD',
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
