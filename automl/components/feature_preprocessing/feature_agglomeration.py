import numpy as np
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenInClause, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm

from automl.util.common import resolve_factor


class FeatureAgglomerationComponent(PreprocessingAlgorithm):
    def __init__(self, n_clusters_factor: int = 2,
                 affinity: str = "euclidean",
                 compute_full_tree: str = "auto",
                 linkage: str = "ward",
                 pooling_func: str = "mean",
                 distance_threshold: float = None):
        super().__init__()
        self.n_clusters_factor = n_clusters_factor
        self.affinity = affinity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.pooling_func = pooling_func

    def fit(self, X, y=None):
        from sklearn.cluster import FeatureAgglomeration

        if self.pooling_func == "mean":
            pooling_func = np.mean
        elif self.pooling_func == "median":
            pooling_func = np.median
        elif self.pooling_func == "max":
            pooling_func = np.max
        else:
            raise ValueError('Unknown pooling function \'{}\''.format(self.pooling_func))

        if self.distance_threshold is not None:
            self.n_clusters = None
            self.compute_full_tree = True

        if isinstance(self.n_clusters_factor, int):
            n_clusters = self.n_clusters_factor
        else:
            n_clusters = max(min(resolve_factor(self.n_clusters_factor, X.shape[1]), (X.shape[1] - 1)), 2)

        self.preprocessor = FeatureAgglomeration(n_clusters=n_clusters,
                                                 affinity=self.affinity,
                                                 compute_full_tree=self.compute_full_tree,
                                                 linkage=self.linkage,
                                                 distance_threshold=self.distance_threshold,
                                                 pooling_func=pooling_func)
        self.preprocessor.fit(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FA',
                'name': 'Feature Agglomeration',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_clusters_factor = UniformFloatHyperparameter("n_clusters_factor", 0., 1., default_value=1.)
        affinity = CategoricalHyperparameter("affinity",
                                             ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
                                             default_value="euclidean")
        compute_full_tree = CategoricalHyperparameter("compute_full_tree", [True, False], default_value=True)
        linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average", "single"], default_value="ward")
        pooling_func = CategoricalHyperparameter("pooling_func", ["mean", "median", "max"], default_value="mean")
        distance_threshold = UniformFloatHyperparameter("distance_threshold", 0., 1.0, default_value=None)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_clusters_factor, affinity, compute_full_tree, linkage, distance_threshold, pooling_func])

        distance_thresholdAndNClustersCondition = EqualsCondition(distance_threshold, n_clusters_factor, 1.)
        cs.add_condition(distance_thresholdAndNClustersCondition)

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["l1", "l2", "manhattan", "cosine", "precomputed"]),
            ForbiddenEqualsClause(linkage, "ward"))
        cs.add_forbidden_clause(affinity_and_linkage)

        full_and_n_clusters = ForbiddenAndConjunction(
            ForbiddenEqualsClause(compute_full_tree, False),
            ForbiddenEqualsClause(n_clusters_factor, 1.))
        cs.add_forbidden_clause(full_and_n_clusters)

        return cs
