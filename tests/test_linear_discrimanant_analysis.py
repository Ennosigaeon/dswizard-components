import numpy as np
import sklearn.naive_bayes
import sklearn.svm

from dswizard.components.classification.linear_discriminant_analysis import LinearDiscriminantAnalysis
from tests import base_test
from util.common import resolve_factor


class TestLinearDiscriminantAnalysis(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = LinearDiscriminantAnalysis()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = LinearDiscriminantAnalysis()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['n_components'] = resolve_factor(config['n_components_factor'],
                                                min(X_train.shape[1], len(np.unique(y_train)) - 1), cs_default=0.5)
        del config['n_components_factor']

        expected = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(**config)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)
