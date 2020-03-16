import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class GradiantBoostingClassifier(PredictionAlgorithm):

    def __init__(self,
                 loss: str = "deviance",
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 subsample: float = 1.,
                 criterion: str = "friedman_mse",
                 min_samples_split: float = 0.1,
                 min_samples_leaf: float = 0.1,
                 min_weight_fraction_leaf: float = 0.,
                 max_depth: int = 3,
                 min_impurity_decrease: float = 1.,
                 max_features: float = 0.1,
                 max_leaf_nodes: int = 10000,
                 warm_start: bool = False,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 10000,
                 tol: float = 1e-4,
                 ccp_alpha: float = 0.
                 ):
        super().__init__()
        self.loss = loss,
        self.learning_rate = learning_rate,
        self.n_estimators = n_estimators,
        self.subsample = subsample,
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        from sklearn.ensemble import GradientBoostingClassifier

        self.estimator = GradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol
        )
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradiant Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)}
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        loss = CategoricalHyperparameter("loss", ["deviance", "exponential"], default_value="deviance")
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 3., default_value=0.1)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 10000, default_value=100)
        subsample = UniformFloatHyperparameter("subsample", 0., 5., default_value=1.)
        criterion = CategoricalHyperparameter("criterion", ["friedman_mse", "mse", "mae"], default_value="friedman_mse")
        min_samples_split = UniformFloatHyperparameter("min_samples_split", 0.0001, 0.9999, default_value=0.1)
        min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", 0.0001, 0.9999, default_value=0.1)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0., 1., default_value=0.)
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 1000, default_value=3)
        min_impurity_decrease = UniformFloatHyperparameter("min_impurity_decrease", 0., 1., default_value=1.)
        max_features = UniformFloatHyperparameter("max_features", 0.0001, 0.9999, default_value=0.1)
        max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 2, 10000, default_value=10000)
        warm_start = CategoricalHyperparameter("warm_start", [True, False], default_value=False)
        validation_fraction = UniformFloatHyperparameter("validation_fraction", 0., 1., default_value=0.1)
        n_iter_no_change = UniformIntegerHyperparameter("n_iter_no_change", 2, 10000, default_value=10000)
        tol = UniformFloatHyperparameter("tol", 1e-6, 5., default_value=1e-4)
        ccp_alpha = UniformFloatHyperparameter("ccp_alpha", 0., 5., default_value=0.)

        cs.add_hyperparameters(
            [loss, learning_rate, n_iter_no_change, n_estimators, subsample, criterion, min_samples_split,
             min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, max_features, max_leaf_nodes,
             warm_start, validation_fraction, tol, ccp_alpha])

        return cs
