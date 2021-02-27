from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    Constant, UnParametrizedHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import check_none, resolve_factor, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, \
    HANDLES_MISSING, HANDLES_NOMINAL_CLASS


# TODO does not honour affinity restrictions
class GradientBoostingClassifier(PredictionAlgorithm):
    def __init__(self,
                 loss: str = 'auto',
                 learning_rate: float = 0.1,
                 max_iter: int = 100,
                 min_samples_leaf_factor: int = 20,
                 max_depth_factor: int = None,
                 max_leaf_nodes_factor: int = 31,
                 max_bins: int = 255,
                 l2_regularization: float = 0.,
                 tol: float = 1e-7,
                 scoring: str = 'f1_weighted',
                 n_iter_no_change: int = 10,
                 validation_fraction: float = 0.1,
                 random_state=None):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf_factor = min_samples_leaf_factor
        self.max_depth_factor = max_depth_factor
        self.max_leaf_nodes_factor = max_leaf_nodes_factor
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

        if check_none(self.scoring):
            self.scoring = None

        # Heuristic to set the tree depth
        max_depth = resolve_factor(self.max_depth_factor, n_features, cs_default=1.)
        if max_depth is not None:
            max_depth = max(max_depth, 2)

        l2_regularization = 0. if self.l2_regularization == 1e-07 else self.l2_regularization

        # Heuristic to set the tree width
        if isinstance(self.max_leaf_nodes_factor, int):
            max_leaf_nodes = self.max_leaf_nodes_factor
        else:
            max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, n_samples, default=31, cs_default=1.)
        if max_leaf_nodes is not None:
            max_leaf_nodes = max(max_leaf_nodes, 2)

        # Heuristic to set the tree width
        if isinstance(self.min_samples_leaf_factor, int):
            min_samples_leaf = self.min_samples_leaf_factor
        else:
            min_samples_leaf = resolve_factor(self.min_samples_leaf_factor, n_samples, default=20, cs_default=0.0001)

        n_iter_no_change = None if self.n_iter_no_change == 0 else self.n_iter_no_change

        if self.scoring == 'balanced_accurary':
            self.scoring = 'balanced_accuracy'

        return HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            max_bins=self.max_bins,
            l2_regularization=l2_regularization,
            tol=self.tol,
            scoring=self.scoring,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

    @staticmethod
    def get_properties():
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        loss = Constant("loss", "auto")
        learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=0.01, upper=1, default_value=0.1,
                                                   log=True)
        min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf_factor", 0.0001, 0.25, default_value=0.0001,
                                                      log=True)
        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 1e-5, 2.5, default_value=1.)
        max_leaf_nodes_factor = UniformFloatHyperparameter("max_leaf_nodes_factor", 1e-5, 1., default_value=1.)
        max_iter = Constant("max_iter", 512)
        max_bins = Constant("max_bins", 255)
        l2_regularization = UniformFloatHyperparameter(name="l2_regularization", lower=1e-10, upper=1., log=True,
                                                       default_value=1e-10)
        tol = UnParametrizedHyperparameter(name="tol", value=1e-7)
        scoring = UnParametrizedHyperparameter(name="scoring", value="loss")
        n_iter_no_change = UniformIntegerHyperparameter(name="n_iter_no_change", lower=1, upper=20, default_value=10)
        validation_fraction = UniformFloatHyperparameter(name="validation_fraction", lower=0.01, upper=0.4,
                                                         default_value=0.1)

        cs.add_hyperparameters(
            [loss, learning_rate, min_samples_leaf, max_depth_factor, max_leaf_nodes_factor, max_bins, max_iter,
             l2_regularization, tol, scoring, n_iter_no_change, validation_fraction, ])

        return cs
