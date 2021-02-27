import numpy as np
import sklearn.naive_bayes
import sklearn.svm
from dswizard.components.classification.excluded.mlp_classifier import MLPClassifier

from tests import base_test


class TestMLPClassifier(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = MLPClassifier(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.neural_network.MLPClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = MLPClassifier(random_state=42)
        config: dict = self.get_config(actual)
        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['hidden_layer_sizes'] = (config['layer_1_size'], config['layer_2_size'])
        del config['layer_1_size']
        del config['layer_2_size']
        print(config)

        expected = sklearn.neural_network.MLPClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)
