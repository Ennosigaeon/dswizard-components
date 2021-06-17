import os
from collections import OrderedDict
from typing import Dict, Type, Optional, List

from dswizard.components.base import PredictionAlgorithm, find_components, ComponentChoice, PredictionMixin, \
    EstimatorComponent

classifier_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__, classifier_directory, PredictionAlgorithm)


class ClassifierChoice(ComponentChoice, PredictionMixin):

    def __init__(self, defaults: Optional[List[str]] = None, new_params: Dict = None):
        if defaults is None:
            defaults = ['random_forest', 'liblinear_svc', 'sgd', 'libsvm_svc']
        super().__init__('classifier_choice', defaults, new_params)

    def get_components(self) -> Dict[str, Type[EstimatorComponent]]:
        components = OrderedDict()
        components.update(_classifiers)
        return components

    def fit(self, X, y, **kwargs):
        if kwargs is None:
            kwargs = {}
        return self.estimator_.fit(X, y, **kwargs)

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def estimator_supports_iterative_fit(self):
        return hasattr(self.estimator_, 'iterative_fit')

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        if fit_params is None:
            fit_params = {}
        return self.estimator_.iterative_fit(X, y, n_iter=n_iter, **fit_params)

    def configuration_fully_fitted(self):
        return self.estimator_.configuration_fully_fitted()
