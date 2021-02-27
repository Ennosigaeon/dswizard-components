from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


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

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            algorithm=self.algorithm,
            weights=self.weights,
            n_neighbors=self.n_neighbors,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric
        )

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'KN',
                'name': 'KNeighbors Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter("n_neighbors", 1, 70, default_value=5)
        weights = CategoricalHyperparameter("weights", ["uniform", "distance"], default_value="uniform")
        algorithm = CategoricalHyperparameter("algorithm", ["ball_tree", "kd_tree", "brute", "auto"],
                                              default_value="auto")
        leaf_size = UniformIntegerHyperparameter("leaf_size", 1, 100, default_value=30)
        p = UniformIntegerHyperparameter("p", 1, 5, default_value=2)
        metric = CategoricalHyperparameter("metric", ["minkowski", "euclidean", "manhattan", "chebyshev"],
                                           default_value="minkowski")

        cs.add_hyperparameters([n_neighbors, weights, algorithm, leaf_size, p, metric])

        return cs
