from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import resolve_factor, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, \
    HANDLES_MISSING, HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class DecisionTree(PredictionAlgorithm):

    def __init__(self,
                 criterion: str = "gini",
                 splitter: str = "best",
                 max_depth_factor: float = None,
                 min_samples_split_factor: int = 2,
                 min_samples_leaf_factor: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 max_features_factor: float = None,
                 random_state=None,
                 max_leaf_nodes_factor: int = None,
                 min_impurity_decrease: float = 0.,
                 class_weight=None,
                 ccp_alpha: float = 0.
                 ):
        super().__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_features_factor = max_features_factor
        self.max_depth_factor = max_depth_factor
        self.min_samples_split_factor = min_samples_split_factor
        self.min_samples_leaf_factor = min_samples_leaf_factor
        self.max_leaf_nodes_factor = max_leaf_nodes_factor
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1])
        self.estimator.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.estimator.classes_
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.tree import DecisionTreeClassifier

        # Heuristic to set the tree depth
        max_depth = resolve_factor(self.max_depth_factor, n_features, cs_default=1.)
        if max_depth is not None:
            max_depth = max(max_depth, 2)

        # Heuristic to set the tree width
        max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, n_samples, cs_default=1.)
        if max_leaf_nodes is not None:
            max_leaf_nodes = max(max_leaf_nodes, 2)

        # Heuristic to set max features
        max_features = resolve_factor(self.max_features_factor, n_features, cs_default=1., default=None)
        if max_features is not None:
            max_features = max(max_features, 1)

        # Heuristic to set min_samples_split
        min_samples_split = resolve_factor(self.min_samples_split_factor, n_samples, default=2, cs_default=0.0001)
        if min_samples_split is not None:
            min_samples_split = max(min_samples_split, 2)

        # Heuristic to set min_samples_leaf
        min_samples_leaf = resolve_factor(self.min_samples_leaf_factor, n_samples, default=1, cs_default=0.0001)
        if min_samples_leaf is not None:
            min_samples_leaf = max(min_samples_leaf, 1)

        return DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 0, 2, default_value=1.)
        min_samples_split = UniformFloatHyperparameter("min_samples_split_factor", 1e-7, 0.25, default_value=0.0001)
        min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf_factor", 1e-7, 0.25, default_value=0.0001)
        min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
        max_features = UnParametrizedHyperparameter('max_features_factor', 1.0)
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

        cs.add_hyperparameters(
            [criterion, max_features, max_depth_factor, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
             min_impurity_decrease])

        return cs
