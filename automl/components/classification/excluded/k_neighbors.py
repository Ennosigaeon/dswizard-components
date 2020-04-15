from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.util import convert_multioutput_multiclass_to_multilabel
import numpy as np


class KNeighborsClassifier(PredictionAlgorithm):

    def __init__(self,
                 n_neighbors: int = 5,
                 weights: str = "uniform",
                 algorithm: str = "auto",
                 leaf_size: int = 30,
                 p: int = 2,
                 metric: str = "minkowski",
                 random_state=None
                 ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier
        self.estimator = KNeighborsClassifier(
            algorithm=self.algorithm,
            weights=self.weights,
            n_neighbors=self.n_neighbors,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric
        )
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KN',
                'name': 'KNeighbors Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)}
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter("n_neighbors", 1, 70, default_value=5)
        weights = CategoricalHyperparameter("weights", ["uniform", "distance"], default_value="uniform")
        algorithm = CategoricalHyperparameter("algorithm", ["ball_tree", "kd_tree", "brute", "auto"], default_value="auto")
        leaf_size = UniformIntegerHyperparameter("leaf_size", 1, 100, default_value=30)
        p = UniformIntegerHyperparameter("p", 1, 5, default_value=2)
        metric = CategoricalHyperparameter("metric", ["minkowski", "euclidean", "manhattan", "chebyshev"],
                                           default_value="minkowski")

        cs.add_hyperparameters([n_neighbors, weights, algorithm, leaf_size, p, metric])

        return cs
