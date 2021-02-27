from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, convert_multioutput_multiclass_to_multilabel


class QuadraticDiscriminantAnalysis(PredictionAlgorithm):

    def __init__(self,
                 reg_param: float = 0.,
                 store_covariance: bool = False,
                 tol: float = 1e-4
                 ):
        super().__init__()
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        return QuadraticDiscriminantAnalysis(
            reg_param=self.reg_param,
            store_covariance=self.store_covariance,
            tol=self.tol
        )

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'QDA',
                'name': 'Quadratic Discriminant Analysis',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        reg_param = UniformFloatHyperparameter("reg_param", 0., 1.5, default_value=0.)
        store_covariance = CategoricalHyperparameter("store_covariance", [True, False], default_value=False)
        tol = UniformFloatHyperparameter("tol", 1e-5, 10.0, default_value=1e-4)

        cs.add_hyperparameters(
            [reg_param, store_covariance, tol])

        return cs
