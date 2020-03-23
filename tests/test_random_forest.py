import numpy as np
import sklearn.naive_bayes
import sklearn.svm

from automl.components.classification.random_forest import RandomForest
from tests import base_test


class TestRandomForest(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RandomForest(random_state=42)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.ensemble.RandomForestClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RandomForest(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['max_features'] = int(X_train.shape[1] ** float(config['max_features']))

        expected = sklearn.ensemble.RandomForestClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)
