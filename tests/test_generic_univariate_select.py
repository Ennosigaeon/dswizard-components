import numpy as np
import sklearn

from automl.components.feature_preprocessing.test.generic_univariate_select import GenericUnivariateSelectComponent
from tests import base_test


class TestGenericUnivariateSelectComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GenericUnivariateSelectComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.feature_selection.GenericUnivariateSelect()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GenericUnivariateSelectComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.feature_selection.GenericUnivariateSelect(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)
