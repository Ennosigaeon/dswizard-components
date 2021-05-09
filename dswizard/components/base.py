import importlib
import inspect
import pkgutil
import sys
from abc import ABC
from collections import OrderedDict
from typing import Type, Dict, Optional, List

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

from dswizard.components.meta_features import MetaFeaturesDict
from dswizard.components.util import HANDLES_NOMINAL, HANDLES_NUMERIC, HANDLES_MISSING, HANDLES_MULTICLASS, \
    HANDLES_NOMINAL_CLASS


def find_components(package: str, directory: str, base_class: Type) -> Dict[str, Type]:
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class MetaData:
    @staticmethod
    def get_properties() -> dict:
        """Get the properties of the underlying algorithm.

        Find more information at :ref:`get_properties`

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(**kwargs) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Returns
        -------
        Configspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def set_hyperparameters(self, configuration, init_params=None) -> 'MetaData':
        raise NotImplementedError()

    @classmethod
    def name(cls, short: bool = False) -> str:
        if short:
            return cls.__qualname__
        else:
            return '.'.join([cls.__module__, cls.__qualname__])


# noinspection PyPep8Naming
class PredictionMixin(ClassifierMixin):

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,) or shape = (n_samples, n_labels)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        raise NotImplementedError()


# noinspection PyPep8Naming
class EstimatorComponent(BaseEstimator, MetaData, ABC):

    def __init__(self, estimator: Optional[BaseEstimator] = None):
        self.estimator: Optional[BaseEstimator] = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EstimatorComponent':
        """The fit function calls the fit function of the underlying
        scikit-learn model and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,) or shape = (n_sample, n_labels)

        Returns
        -------
        self : returns an instance of self.
            Targets

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying scikit-learn model and returns the transformed array.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        X : array
            Return the transformed training data

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def serialize(self):
        # TODO kwargs for __init__ not persisted
        cls = self.__class__
        return '.'.join([cls.__module__, cls.__qualname__])

    def set_hyperparameters(self, configuration: dict = None, init_params=None) -> 'EstimatorComponent':
        if configuration is None:
            configuration = self.get_hyperparameter_search_space().get_default_configuration().get_dictionary()

        for param, value in configuration.items():
            if not hasattr(self, param) and param != 'random_state':
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
                                     (param, str(self)))
                setattr(self, param, value)

        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        """Transforms this Component to a standard sklearn component if possible"""
        return self

    def __str__(self):
        cls = self.__class__
        return '.'.join([cls.__module__, cls.__qualname__])

    def __repr__(self, **kwargs):
        return self.__class__.__qualname__


# noinspection PyPep8Naming
class PredictionAlgorithm(EstimatorComponent, PredictionMixin, ABC):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    def __init__(self):
        super().__init__()
        self.properties: Optional[Dict] = None
        # TODO generalize for other learning tasks
        self._estimator_type = "classifier"
        self.classes_ = None

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator

    def fit(self, X, Y):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1])
        self.estimator.fit(X, Y)
        self.classes_ = self.estimator.classes_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noinspection PyTypeChecker
        X: np.ndarray = check_array(X)
        try:
            # add class probabilities as a synthetic feature
            X_transformed = np.hstack((X, self.estimator.predict_proba(X)))
        except AttributeError:
            # Some classifiers do not implement predict_proba
            X_transformed = X

        # add class prediction as a synthetic feature
        # noinspection PyUnresolvedReferences
        y = np.reshape(self.estimator.predict(X), (-1, 1))
        try:
            y = y.astype(float)
        except ValueError:
            pass
        X_transformed = np.hstack((X_transformed, y))

        return X_transformed

    def predict(self, X):
        if self.estimator is None:
            raise ValueError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise ValueError()
        return self.estimator.predict_proba(X)


class PreprocessingAlgorithm(EstimatorComponent, ABC):
    """Provide an abstract interface for preprocessing algorithms in auto-sklearn.

    See :ref:`extending` for more information."""

    def fit(self, X, Y):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1])
        self.estimator.fit(X, Y)
        return self

    def transform(self, X):
        if self.estimator is None:
            raise ValueError()
        return self.estimator.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        return cs


class NoopComponent(EstimatorComponent):

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    @staticmethod
    def get_properties() -> dict:
        return {'shortname': 'noop',
                'name': 'No Operation',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs) -> ConfigurationSpace:
        return ConfigurationSpace()


# noinspection PyPep8Naming
class ComponentChoice(EstimatorComponent):

    def __init__(self, defaults: List[str], estimator: Optional[BaseEstimator] = None, new_params: Dict = None):
        super().__init__(estimator)
        self.defaults = defaults
        self.new_params = new_params
        self.configuration_space_: Optional[ConfigurationSpace] = None

    def get_components(self) -> Dict[str, Type[EstimatorComponent]]:
        raise NotImplementedError()

    def get_available_components(self, mf: MetaFeaturesDict = None,
                                 include: List = None,
                                 exclude: List = None) -> Dict[str, Type[EstimatorComponent]]:
        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == type(self) or hasattr(entry, 'get_components'):
                continue

            # Basic check if component is compatible with data set
            if mf is not None:
                props = entry.get_properties()
                if mf['nr_cat'] > 0 and not props[HANDLES_NOMINAL]:
                    continue
                if mf['nr_num'] > 0 and not props[HANDLES_NUMERIC]:
                    continue
                if mf['nr_missing_values'] > 0 and not props[HANDLES_MISSING]:
                    continue

            components_dict[name] = entry

        return components_dict

    def set_hyperparameters(self, configuration: dict = None, init_params=None) -> 'ComponentChoice':
        if configuration is None:
            raise ValueError('Default hyperparameters not available for ComponentChoice')
        new_params = {}

        choice = configuration['__choice__']
        for param, value in configuration.items():
            if param == '__choice__':
                continue

            param = param.replace(choice, '').replace(':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, '').replace(':', '')
                new_params[param] = value

        self.new_params = new_params
        self.estimator = self.get_components()[choice]().set_hyperparameters(new_params)

        return self

    def get_hyperparameter_search_space(self, mf: MetaFeaturesDict = None,
                                        default=None,
                                        include=None,
                                        exclude=None,
                                        **kwargs):
        if include is not None and exclude is not None:
            raise ValueError("The arguments include and exclude cannot be used together.")
        cs = ConfigurationSpace()

        # Compile a list of all estimator objects for this problem
        available_estimators = self.get_available_components(mf=mf, include=include, exclude=exclude)

        if len(available_estimators) == 0:
            raise ValueError("No classifiers found")

        if default is None:
            for default_ in self.defaults:
                if default_ in available_estimators:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        estimator = CategoricalHyperparameter('__choice__', list(available_estimators.keys()), default_value=default)
        cs.add_hyperparameter(estimator)
        for estimator_name in available_estimators.keys():
            estimator_configuration_space = available_estimators[estimator_name].get_hyperparameter_search_space()
            parent_hyperparameter = {'parent': estimator, 'value': estimator_name}
            cs.add_configuration_space(estimator_name, estimator_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space_ = cs
        return cs

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.transform(X)

    @staticmethod
    def get_properties() -> Dict:
        return {}
