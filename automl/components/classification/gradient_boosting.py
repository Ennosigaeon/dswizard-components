import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant, CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none

from automl.util.common import resolve_factor


class GradientBoostingClassifier(PredictionAlgorithm):
    def __init__(self,
                 loss: str = 'auto',
                 learning_rate: float = 0.1,
                 max_iter: int = 100,
                 min_samples_leaf: int = 20,
                 max_depth_factor: int = None,
                 max_leaf_nodes_factor: int = 31,
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
        self.max_depth_factor = max_depth_factor
        self.max_leaf_nodes_factor = max_leaf_nodes_factor
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, Y):
        from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

        if check_none(self.scoring):
            self.scoring = None

        # Heuristic to set the tree depth
        max_depth = resolve_factor(self.max_depth_factor, X.shape[1])
        if max_depth is not None:
            max_depth = max(max_depth, 2)

        # Heuristic to set the tree width
        if isinstance(self.max_leaf_nodes_factor, int):
            max_leaf_nodes = self.max_leaf_nodes_factor
        else:
            max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, X.shape[0])
        if max_leaf_nodes is not None:
            max_leaf_nodes = max(max_leaf_nodes, 2)

        # Heuristic to set the tree width
        if isinstance(self.min_samples_leaf, int):
            min_samples_leaf = self.min_samples_leaf
        else:
            min_samples_leaf = resolve_factor(self.min_samples_leaf, X.shape[0])

        self.estimator = HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
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
        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 1e-5, 2.5, default_value=1.)
        max_iter = UniformIntegerHyperparameter("max_iter", 0, 1000, default_value=100)
        max_leaf_nodes_factor = UniformFloatHyperparameter("max_leaf_nodes_factor", 1e-5, 1., default_value=1.)
        min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", 0.0001, 0.5, default_value=0.0001)
        l2_regularization = UniformFloatHyperparameter(name="l2_regularization", lower=1e-7, upper=10.,
                                                       default_value=1e-7, log=True)
        max_bins = UniformIntegerHyperparameter("max_bins", 5, 255, default_value=255)
        tol = UniformFloatHyperparameter("tol", 0., 0.25, default_value=1e-7)
        scoring = CategoricalHyperparameter("scoring",
                                            ["accuracy", "balanced_accuracy", "average_precision", "neg_brier_score",
                                             "f1", "f1_micro", "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
                                             "precision", "precision_micro", "precision_macro", "precision_weighted",
                                             "precision_samples", "recall", "recall_micro", "recall_macro",
                                             "recall_weighted", "recall_samples", "jaccard", "jaccard_micro",
                                             "jaccard_macro", "jaccard_weighted", "jaccard_samples", "roc_auc",
                                             "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted",
                                             "roc_auc_ovo_weighted"], default_value="f1_weighted")
        n_iter_no_change = UniformIntegerHyperparameter(name="n_iter_no_change", lower=0, upper=100, default_value=0)
        validation_fraction = UniformFloatHyperparameter(name="validation_fraction", lower=0.001, upper=1.0,
                                                         default_value=0.1)

        cs.add_hyperparameters([loss, learning_rate, max_iter, min_samples_leaf, max_leaf_nodes_factor, max_bins,
                                l2_regularization, tol, scoring, n_iter_no_change, validation_fraction,
                                max_depth_factor])

        return cs
