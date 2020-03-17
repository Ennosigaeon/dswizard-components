from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from automl.components.base import PreprocessingAlgorithm


class ImputationComponent(PreprocessingAlgorithm):
    def __init__(self, strategy: str = 'mean', copy: bool = True, add_indicator: bool = False):
        super().__init__()
        self.strategy = strategy
        self.copy = copy
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        import sklearn.impute
        self.preprocessor = sklearn.impute.SimpleImputer(strategy=self.strategy, copy=self.copy, add_indicator=self.add_indicator)
        self.preprocessor = self.preprocessor.fit(X)
        return self

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
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter("strategy", ["mean", "median", "most_frequent"], default_value="mean")
        copy = CategoricalHyperparameter("copy", [True,False], default_value=True)
        add_indicator = CategoricalHyperparameter("add_indicator", [True,False], default_value=False)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy,copy,add_indicator)
        return cs
