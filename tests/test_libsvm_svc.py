import numpy as np
import sklearn.naive_bayes
import sklearn.svm

from dswizard.components.classification.libsvm_svc import LibSVM_SVC
from tests import base_test


class TestLibSVM_SVC(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = LibSVM_SVC(random_state=42)
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.svm.SVC(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['prediction']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = LibSVM_SVC(random_state=42)
        config = self.get_config(actual, seed=0)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.svm.SVC(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['prediction']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)
