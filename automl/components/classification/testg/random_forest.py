from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none, check_for_bool
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class RandomForest(PredictionAlgorithm):
    def __init__(self,
                 n_estimators: int = 10,
                 criterion: str = 'gini',
                 max_features: int = 'auto',
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 bootstrap: bool = True,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.,
                 oob_score: bool = False,
                 warm_start: bool = False,
                 ccp_alpha: float = 0.1,
                 max_samples: float = 1.,
                 random_state=None,
                 class_weight=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestClassifier

        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)

        if self.max_features not in ("sqrt", "log2", "auto"):
            max_features = int(X.shape[1] ** float(self.max_features))
        else:
            max_features = self.max_features

        self.bootstrap = check_for_bool(self.bootstrap)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        if self.max_depth == 1:
            self.max_depth = None

        if self.max_leaf_nodes == 1:
            self.max_leaf_nodes = None

        # initial fit of only increment trees
        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            class_weight=self.class_weight,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            max_samples=self.max_samples,
            ccp_alpha=self.ccp_alpha)
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
        return {'shortname': 'RF',
                'name': 'Random Forest Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 4000, default_value=100)
        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter("max_features", 0., 1., default_value=0.5)
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 50, default_value=10)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 60, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 60, default_value=1)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0., 1., default_value=0.)
        max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 1, 100, default_value=10)
        min_impurity_decrease = UniformFloatHyperparameter('min_impurity_decrease', 0., 0.75, default_value=0.)
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=True)
        oob_score = CategoricalHyperparameter("oob_score", [True,False], default_value=False)
        warm_start = CategoricalHyperparameter("warm_start", [True,False], default_value=False)
        ccp_alpha = UniformFloatHyperparameter("ccp_alpha", 0., 1., default_value=0.1)
        max_samples = UniformFloatHyperparameter("max_samples", 0., 1., default_value=1.)
        cs.add_hyperparameters([n_estimators, criterion, max_features, max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes, bootstrap, min_impurity_decrease, oob_score, warm_start, ccp_alpha, max_samples])

        return cs
