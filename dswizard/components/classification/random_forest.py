from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class RandomForest(PredictionAlgorithm):
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_features: int = 'auto',
                 max_depth: float = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 bootstrap: bool = True,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.,
                 oob_score: bool = False,
                 ccp_alpha: float = 0.0,
                 max_samples: float = None,
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
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def fit(self, X, y, sample_weight=None):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1])
        self.estimator.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.estimator.classes_
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.ensemble import RandomForestClassifier

        if self.max_features == 0.5:
            max_features = 'auto'
        elif self.max_features not in ("sqrt", "log2", "auto"):
            max_features = int(n_features ** float(self.max_features))
        else:
            max_features = self.max_features

        max_samples = None if self.max_samples == 0.99 else self.max_samples

        # initial fit of only increment trees
        return RandomForestClassifier(
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
            max_samples=max_samples,
            n_jobs=1,
            ccp_alpha=self.ccp_alpha)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'RF',
                'name': 'Random Forest Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        n_estimators = UnParametrizedHyperparameter("n_estimators", 512)

        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
        # The maximum number of features used in the forest is calculated as m^max_features, where
        # m is the total number of features, and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
        # corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter("max_features", 0., 1.0, default_value=0.5)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
        bootstrap = CategoricalHyperparameter("bootstrap", [True, False], default_value=True)

        cs.add_hyperparameters(
            [n_estimators, criterion, max_features, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
             bootstrap, min_impurity_decrease])

        return cs
