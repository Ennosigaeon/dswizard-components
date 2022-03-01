from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import _check_feature_names_in

from dswizard.components import util
from dswizard.components.base import PreprocessingAlgorithm, EstimatorComponent, HasChildComponents, PredictionAlgorithm
from dswizard.components.meta_features import MetaFeaturesDict
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS


def monkey_patch_get_feature_names_out():
    # Some transformers do not implement get_feature_names_out, monkey-patch it in
    if 'get_feature_names_out_patched' not in globals():
        func = lambda est, input_features=None: _check_feature_names_in(est, input_features)
        SimpleImputer.get_feature_names_out = func
        OrdinalEncoder.get_feature_names_out = func

        # add marker to globals to prevent second execution
        # noinspection PyGlobalUndefined
        global get_feature_names_out_patched
        get_feature_names_out_patched = True


monkey_patch_get_feature_names_out()


class StackingEstimator(PreprocessingAlgorithm, PredictionAlgorithm):
    """StackingEstimator

    A shallow wrapper around a classification algorithm to implement the transform method. Allows stacking of
    arbitrary classification algorithms in a pipelines.

    Read more in the :ref:`User Guide`.

    Parameters
    ----------
    estimator : PredictionMixin
        An instance implementing PredictionMixin

    Attributes
    ----------
    estimator : PredictionMixin
        The wrapped PredicitionMixin instance

    See also
    --------
    PredicitionMixin
        Instances that can be stacked using this class.

    References
    ----------
    .. [1] `Stacked Generalization
        <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.133.8090&rep=rep1&type=pdf>`_
    """

    def __init__(self, estimator: PredictionAlgorithm):
        super().__init__(estimator.component_name_)
        self.estimator_ = estimator

    @staticmethod
    def get_properties() -> Dict:
        return {'shortname': 'se',
                'name': 'Stacking Estimator',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True
                }

    def fit(self, X, y=None):
        self.estimator_.fit(X, y)
        return self

    def transform(self, X, *args) -> np.ndarray:
        # add class probabilities as a synthetic feature
        # noinspection PyUnresolvedReferences
        try:
            X_transformed = np.hstack((X, self.estimator_.predict_proba(X)))
        except AttributeError:
            X_transformed = X

        # add class prediction as a synthetic feature
        # noinspection PyUnresolvedReferences
        X_transformed = np.hstack((X_transformed, np.reshape(self.estimator_.predict(X), (-1, 1))))

        return X_transformed

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def get_params(self, deep=True):
        return {'estimator': self.estimator_}

    def set_params(self, **params):
        self.estimator_ = params['estimator']

    def __repr__(self, N_CHAR_MAX=700):
        return f'Stacking({self.estimator_.__repr__()})'


class ColumnTransformerComponent(ColumnTransformer, PreprocessingAlgorithm, HasChildComponents):

    def __init__(self, transformers: List[Tuple[str, EstimatorComponent, Any]], **kwargs):
        self.args = {
            'transformers': [(label, util.serialize(comp), columns) for label, comp, columns in transformers],
            **kwargs
        }
        super().__init__(transformers, **kwargs)

    @staticmethod
    def get_properties() -> Dict:
        return {'shortname': 'ct',
                'name': 'Column Transformer',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def deserialize(transformers: List[Dict[str, Any]], **kwargs) -> 'ColumnTransformerComponent':
        transformers_ = []
        for name, value, columns in transformers:
            transformers_.append((name, util.deserialize(**value), columns))
        return ColumnTransformerComponent(transformers_, **kwargs)

    def get_hyperparameter_search_space(self, mf: Optional[MetaFeaturesDict] = None):
        return self.get_child_hyperparameter_search_space([(name, comp) for name, comp, _ in self.transformers], mf)

    def set_hyperparameters(self, configuration: Dict = None, init_params=None) -> 'ColumnTransformerComponent':
        self.set_child_hyperparameters([(name, comp) for name, comp, _ in self.transformers], configuration,
                                       init_params)
        return self

    def fit_transform(self, X, y=None):
        res = super(ColumnTransformerComponent, self).fit_transform(X, y)
        self.estimator_ = self
        return res


class FeatureUnionComponent(FeatureUnion, PreprocessingAlgorithm, HasChildComponents):

    def __init__(self, transformer_list: List[Tuple[str, EstimatorComponent]], **kwargs):
        self.args = {
            'transformer_list': [(label, util.serialize(comp)) for label, comp in transformer_list],
            **kwargs
        }
        super().__init__(transformer_list, **kwargs)

    @staticmethod
    def deserialize(transformer_list: List[Dict[str, Any]], **kwargs) -> 'FeatureUnionComponent':
        transformer_list_ = []
        for name, value in transformer_list:
            transformer_list_.append((name, util.deserialize(**value)))
        return FeatureUnionComponent(transformer_list_, **kwargs)

    @staticmethod
    def get_properties() -> Dict:
        return {'shortname': 'parallel',
                'name': 'Feature Union',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True
                }

    def get_hyperparameter_search_space(self, mf: Optional[MetaFeaturesDict] = None):
        return self.get_child_hyperparameter_search_space(self.transformer_list, mf)

    def set_hyperparameters(self, configuration: Dict = None, init_params=None) -> 'FeatureUnionComponent':
        self.set_child_hyperparameters(self.transformer_list, configuration, init_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        res = super(FeatureUnionComponent, self).fit_transform(X, y, **fit_params)
        self.estimator_ = self
        return res
