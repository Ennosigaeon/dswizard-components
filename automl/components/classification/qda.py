import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class SVCClassifier(PredictionAlgorithm):

    def __init__(self,
                 reg_param: float = None,
                 store_covariance: bool = True,
                 tol: float = 1e-4
                 ):
        super().__init__()
        self.reg_param = reg_param,
        self.store_covariance = store_covariance,
        self.tol = tol

    def fit(self, X, y):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        self.estimator = QuadraticDiscriminantAnalysis(
            reg_param=self.reg_param,
            store_covariance=self.store_covariance,
            tol=self.tol
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
        return {'shortname': 'QDA',
                'name': 'Quadratic Discriminant Analysis',
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

        reg_param = UniformFloatHyperparameter("reg_param", 0., 10., default_value=None)
        store_covariance = CategoricalHyperparameter("store_covariance", [True, False], default_value=True)
        tol = UniformFloatHyperparameter("tol", 1e-5, 4.0, default_value=1e-4)

        cs.add_hyperparameters(
            [reg_param, store_covariance, tol])

        return cs
