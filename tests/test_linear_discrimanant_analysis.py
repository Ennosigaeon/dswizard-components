import numpy as np
import sklearn.naive_bayes
import sklearn.svm

from dswizard.components.classification.linear_discriminant_analysis import LinearDiscriminantAnalysis
from dswizard.components.util import resolve_factor
from tests import base_test


class TestLinearDiscriminantAnalysis(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = LinearDiscriminantAnalysis()
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['prediction']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = LinearDiscriminantAnalysis()
        config = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['n_components'] = resolve_factor(config['n_components_factor'],
                                                min(X_train.shape[1], len(np.unique(y_train)) - 1), cs_default=0.5)
        del config['n_components_factor']

        expected = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(**config)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['prediction']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)
