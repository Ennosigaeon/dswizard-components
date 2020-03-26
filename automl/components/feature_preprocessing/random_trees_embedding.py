from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import check_none, check_for_bool

from automl.util.common import resolve_factor


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
                 n_jobs: int = None,
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
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _fit(self, X, Y=None):
        from sklearn.ensemble import RandomTreesEmbedding

        self.n_estimators = int(self.n_estimators)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.bootstrap = check_for_bool(self.bootstrap)

        # Heuristic to set the tree depth
        if isinstance(self.max_depth_factor, int):
            max_depth = self.max_depth_factor
        else:
            max_depth = resolve_factor(self.max_depth_factor, X.shape[1])

        # Heuristic to set the tree width
        max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, X.shape[0])

        # Heuristic to set the tree width
        if isinstance(self.min_samples_leaf_factor, int):
            min_samples_leaf = self.min_samples_leaf_factor
        else:
            min_samples_leaf= resolve_factor(self.min_samples_leaf_factor, X.shape[0])

        # Heuristic to set the tree width
        if isinstance(self.min_samples_split_factor, int):
            min_samples_split = self.min_samples_split_factor
        else:
            min_samples_split = resolve_factor(self.min_samples_split_factor, X.shape[0])

        self.preprocessor = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            sparse_output=False,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.preprocessor.fit(X, Y)
        return self

    def fit(self, X, y):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    def fit(self, X, y):
        self._fit(X)
        return self


    def fit_transform(self, X, y=None):
        return self._fit(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RandomTreesEmbedding',
                'name': 'Random Trees Embedding',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (SPARSE, SIGNED_DATA)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter(name="n_estimators", lower=10, upper=400, default_value=10)
        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 1e-5, 1., default_value=1.)
        min_samples_split_factor = UniformFloatHyperparameter("min_samples_split_factor", 0.0001, 0.5, default_value=0.0001)
        min_samples_leaf_factor = UniformFloatHyperparameter("min_samples_leaf_factor", 0.0001, 0.5, default_value=0.0001)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0., 0.5, default_value=0.)
        max_leaf_nodes_factor = UniformFloatHyperparameter("max_leaf_nodes_factor", 1e-5, 1., default_value=1.)
        min_impurity_decrease = UniformFloatHyperparameter('min_impurity_decrease', 0., 0.75, default_value=0.)

        cs.add_hyperparameters([n_estimators, max_depth_factor, min_samples_split_factor, min_samples_leaf_factor,
                                min_weight_fraction_leaf, max_leaf_nodes_factor, min_impurity_decrease])
        return cs
