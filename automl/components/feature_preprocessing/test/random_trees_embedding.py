from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import check_none, check_for_bool


class RandomTreesEmbeddingComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_estimators: int = 10,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 max_leaf_nodes: int = None,
                 min_impurity_decrease: float = 0.,
                 random_state=None,
                 n_jobs: int = None,
                 bootstrap: bool = True,
                 sparse_output: bool = False
                 ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.sparse_output = sparse_output

    def _fit(self, X, Y=None):
        from sklearn.ensemble import RandomTreesEmbedding

        self.n_estimators = int(self.n_estimators)
        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.bootstrap = check_for_bool(self.bootstrap)

        if self.max_depth == 1:
            self.max_depth = None

        if self.max_leaf_nodes == 1:
            self.max_leaf_nodes = None

        self.preprocessor = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            sparse_output=self.sparse_output,
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
        max_depth = UniformIntegerHyperparameter(name="max_depth", lower=2, upper=50, default_value=5)
        n_jobs = UnParametrizedHyperparameter("n_jobs", "None")
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 60, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 60, default_value=1)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0., 1., default_value=0.)
        max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 1, 100, default_value=1)
        min_impurity_decrease = UniformFloatHyperparameter('min_impurity_decrease', 0., 0.75, default_value=0.)

        cs.add_hyperparameters([n_estimators, n_jobs, max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease])
        return cs
