import numpy as np
from automl.components.feature_preprocessing.binarizer import BinarizerComponent
from sklearn.preprocessing import Binarizer

from tests import base_test


class TestBinarizer(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = BinarizerComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = Binarizer(copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = BinarizerComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        variance = np.mean(np.var(X_train))
        config['threshold'] = max(0., int(np.round(variance * config['threshold_factor'], 0)))
        del config['threshold_factor']

        expected = Binarizer(**config, copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
