import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class DecisionTree(PredictionAlgorithm):

    def __init__(self,
                 criterion: str = "gini",
                 max_depth_factor: float = 1,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 max_features: float = None,
                 random_state=None,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.,
                 class_weight=None,
                 ):
        super().__init__()
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        from sklearn.tree import DecisionTreeClassifier

        # Heuristic to set the tree depth
        if check_none(self.max_depth_factor):
            max_depth = self.max_depth_factor = None
        else:
            num_features = X.shape[1]
            max_depth = max(1, int(np.round(self.max_depth_factor * num_features, 0)))
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None

        self.estimator = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            random_state=self.random_state)
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
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

        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
        max_depth_factor = UniformFloatHyperparameter('max_depth_factor', 0., 2., default_value=0.5)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
        max_features = UnParametrizedHyperparameter('max_features', 1.0)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

        cs.add_hyperparameters([criterion, max_features, max_depth_factor,
                                min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                min_impurity_decrease])

        return cs
