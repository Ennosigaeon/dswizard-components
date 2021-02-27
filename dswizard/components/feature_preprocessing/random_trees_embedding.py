from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, CategoricalHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import check_for_bool, resolve_factor, HANDLES_NOMINAL_CLASS, HANDLES_MISSING, \
    HANDLES_NOMINAL, HANDLES_NUMERIC, HANDLES_MULTICLASS


class RandomTreesEmbeddingComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth_factor: int = 5,
                 min_samples_split_factor: int = 2,
                 min_samples_leaf_factor: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 max_leaf_nodes_factor: int = None,
                 min_impurity_decrease: float = 0.,
                 random_state=None,
                 bootstrap: bool = True
                 ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth_factor = max_depth_factor
        self.min_samples_split_factor = min_samples_split_factor
        self.min_samples_leaf_factor = min_samples_leaf_factor
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes_factor = max_leaf_nodes_factor
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.ensemble import RandomTreesEmbedding

        self.n_estimators = int(self.n_estimators)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.bootstrap = check_for_bool(self.bootstrap)

        # Heuristic to set the tree depth
        if isinstance(self.max_depth_factor, int):
            max_depth = self.max_depth_factor
        else:
            max_depth = resolve_factor(self.max_depth_factor, n_features, default=5, cs_default=1.)
        if max_depth is not None:
            max_depth = max(max_depth, 2)

        # Heuristic to set the tree width
        max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, n_samples, cs_default=1.)
        if max_leaf_nodes is not None:
            max_leaf_nodes = max(max_leaf_nodes, 2)

        # Heuristic to set max features
        min_samples_split = resolve_factor(self.min_samples_split_factor, n_samples, default=2, cs_default=0.0001)
        if min_samples_split is not None:
            min_samples_split = max(min_samples_split, 2)

        # Heuristic to set max features
        min_samples_leaf = resolve_factor(self.min_samples_leaf_factor, n_samples, default=1, cs_default=0.0001)
        if min_samples_leaf is not None:
            min_samples_leaf = max(min_samples_leaf, 1)

        return RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            sparse_output=False,
            n_jobs=1,
            random_state=self.random_state
        )

    @staticmethod
    def get_properties():
        return {'shortname': 'RandomTreesEmbedding',
                'name': 'Random Trees Embedding',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter(name="n_estimators", lower=10, upper=400, default_value=100)
        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 1e-5, 2.5, default_value=1.)
        min_samples_split_factor = UniformFloatHyperparameter("min_samples_split_factor", 0.0001, 0.5,
                                                              default_value=0.0001)
        min_samples_leaf_factor = UniformFloatHyperparameter("min_samples_leaf_factor", 0.0001, 0.5,
                                                             default_value=0.0001)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0., 0.5, default_value=0.)
        bootstrap = CategoricalHyperparameter('bootstrap', [True, False])

        cs.add_hyperparameters([n_estimators, max_depth_factor, min_samples_split_factor, min_samples_leaf_factor,
                                min_weight_fraction_leaf, bootstrap])
        return cs
