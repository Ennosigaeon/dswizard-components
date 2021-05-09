import os
from collections import OrderedDict
from typing import Dict, Type, Optional, List

from sklearn.base import BaseEstimator

from dswizard.components.base import PreprocessingAlgorithm, find_components, ComponentChoice, EstimatorComponent, \
    NoopComponent

classifier_directory = os.path.split(__file__)[0]
_preprocessors = find_components(__package__,
                                 classifier_directory,
                                 PreprocessingAlgorithm)


class FeaturePreprocessorChoice(ComponentChoice):

    def __init__(self, defaults: Optional[List[str]] = None, estimator: Optional[BaseEstimator] = None,
                 new_params: Dict = None):
        if defaults is None:
            defaults = ['no_preprocessing', 'select_percentile', 'pca', 'truncatedSVD']
        super().__init__(defaults, estimator, new_params)

    def get_components(self) -> Dict[str, Type[EstimatorComponent]]:
        components = OrderedDict()
        components['noop'] = NoopComponent
        # noinspection PyTypeChecker
        components.update(_preprocessors)
        return components

    def fit_transform(self, X, y=None):
        return self.estimator.fit(X, y).transform(X)
