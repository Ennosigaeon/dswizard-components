import numpy as np
import sklearn.naive_bayes
import sklearn.svm

from dswizard.components.classification.excluded.qda import QuadraticDiscriminantAnalysis
from tests import base_test


class TestQuadraticDiscriminantAnalysis(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = QuadraticDiscriminantAnalysis()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = QuadraticDiscriminantAnalysis()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(**config)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)
