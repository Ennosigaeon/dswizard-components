from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant, CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none


class GradientBoostingClassifier(PredictionAlgorithm):
    def __init__(self,
                 loss: str = 'auto',
                 learning_rate: float = 0.1,
                 max_iter: int = 100,
                 min_samples_leaf: int = 20,
                 max_depth: int = None,
                 max_leaf_nodes: int = 31,
                 max_bins: int = 255,
                 l2_regularization: float = 0.,
                 tol: float = 1e-7,
                 scoring: str = None,
                 n_iter_no_change: int = None,
                 validation_fraction: float = 0.1,
                 random_state=None):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.tol = tol
        self.max_depth = max_depth
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, Y):
        from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

        if check_none(self.max_depth):
            self.max_depth = None
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        if check_none(self.scoring):
            self.scoring = None

        if self.max_depth == 1:
            self.max_depth = None

        if self.max_leaf_nodes == 1:
            self.max_leaf_nodes = None

        self.estimator = HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            max_bins=self.max_bins,
            l2_regularization=self.l2_regularization,
            tol=self.tol,
            scoring=self.scoring,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

        self.estimator.fit(X, Y)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        loss = Constant("loss", "auto")
        learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=1e-6, upper=1.5, default_value=0.1,
                                                   log=True)
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 50, default_value=1)
        max_iter = UniformIntegerHyperparameter("max_iter", 0, 1000, default_value=100)
        min_samples_leaf = UniformIntegerHyperparameter(name="min_samples_leaf", lower=1, upper=60, default_value=20,
                                                        log=True)
        max_leaf_nodes = UniformIntegerHyperparameter(name="max_leaf_nodes", lower=1, upper=50, default_value=1,
                                                      log=True)
        max_bins = UniformIntegerHyperparameter("max_bins", 5, 255, default_value=255)
        l2_regularization = UniformFloatHyperparameter(name="l2_regularization", lower=1e-7, upper=1.,
                                                       default_value=1e-7, log=True)
        tol = UniformFloatHyperparameter("tol", 0., 0.25, default_value=1e-7)
        scoring = CategoricalHyperparameter("scoring", ["accuracy", "balanced_accuracy", "average_precision", "f1",
                                                        "f1_weighted", "precision", "recall", "roc_auc", "roc_auc_ovr",
                                                        "roc_auc_ovo", "loss"], default_value="loss")
        n_iter_no_change = UniformIntegerHyperparameter(name="n_iter_no_change", lower=1, upper=100, default_value=10)
        validation_fraction = UniformFloatHyperparameter(name="validation_fraction", lower=0.001, upper=0.5,
                                                         default_value=0.1)

        cs.add_hyperparameters([loss, learning_rate, max_iter, min_samples_leaf, max_leaf_nodes, max_bins,
                                l2_regularization, tol, scoring, n_iter_no_change, validation_fraction, max_depth])

        return cs
