from scipy import sparse
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter

from automl.components.base import PreprocessingAlgorithm


class FastICAComponent(PreprocessingAlgorithm):
    def __init__(self):
        super().__init__()
        from sklearn.decomposition import FastICA
        self.preprocessor = FastICA()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_components = UniformIntegerHyperparameter("n_components", 10, 10000, default_value= 100)
        algorithm = CategoricalHyperparameter("algorithm", ["parallel", "deflation"], default_value="parallel")
        whiten = CategoricalHyperparameter("whiten", [True,False], default_value=True)
        fun = CategoricalHyperparameter("fun", ["logcosh", "exp", "cube"], default_value="logcosh")
        max_iter = UniformIntegerHyperparameter("max_iter", 1, 1000, default_value=100)
        tol = UniformFloatHyperparameter("tol", 1e-5, 5., default_value=1.)

        cs.add_hyperparameter(n_components,algorithm, whiten, fun, max_iter, tol)
        return cs

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
