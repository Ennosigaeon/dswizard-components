from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import resolve_factor, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, \
    HANDLES_MISSING, HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class ExtraTreesClassifier(PredictionAlgorithm):
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_features: int = 'auto',
                 max_depth_factor: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 bootstrap: bool = False,
                 max_leaf_nodes_factor: int = None,
                 min_impurity_decrease: float = 0.,
                 ccp_alpha: float = 0.,
                 random_state=None,
                 class_weight=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes_factor = max_leaf_nodes_factor
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1])
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.ensemble import ExtraTreesClassifier

        # Heuristic to set the tree depth
        max_depth = resolve_factor(self.max_depth_factor, n_features, cs_default=1.)
        if max_depth is not None:
            max_depth = max(max_depth, 2)

        # Heuristic to set the tree width
        max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, n_samples, cs_default=1.)
        if max_leaf_nodes is not None:
            max_leaf_nodes = max(max_leaf_nodes, 2)

        max_features = 'auto' if self.max_features == 0.5 else self.max_features
        min_samples_leaf = 1 if self.min_samples_leaf == 0.0001 else self.min_samples_leaf
        min_samples_split = 2 if self.min_samples_split == 0.0001 else self.min_samples_split

        # initial fit of only increment trees
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state,
            class_weight=self.class_weight)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'ET',
                'name': 'Extra Trees Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):

        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 750, default_value=100)
        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")

        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter("max_features", 1e-4, 1., default_value=0.5)

        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 1e-7, 5., default_value=1.)
        min_samples_split = UniformFloatHyperparameter("min_samples_split", 1e-7, 0.5, default_value=0.0001)
        min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", 1e-7, 0.5, default_value=0.0001)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0., 0.5, default_value=0.)
        max_leaf_nodes_factor = UniformFloatHyperparameter("max_leaf_nodes_factor", 1e-7, 1., default_value=1.)
        min_impurity_decrease = UniformFloatHyperparameter('min_impurity_decrease', 0., 0.2, default_value=0.)
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=False)
        ccp_alpha = UniformFloatHyperparameter("ccp_alpha", 0., 1., default_value=0.)

        cs.add_hyperparameters([n_estimators, criterion, max_features, max_depth_factor, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes_factor, bootstrap, min_impurity_decrease, ccp_alpha])
        return cs
