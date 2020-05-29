import numpy as np
import pandas as pd

from automl.components.feature_preprocessing.one_hot_encoding import OneHotEncoderComponent
from tests import base_test


class TestOneHotEncoderComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = OneHotEncoderComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        X_expected = pd.get_dummies(X_test, sparse=False)

        assert np.allclose(X_actual, X_expected)

    def test_categorical(self):
        actual = OneHotEncoderComponent()
        X_before = pd.DataFrame([['Mann', 1], ['Frau', 2], ['Frau', 1]], columns=['Gender', 'Label'])
        y_before = pd.Series([1, 1, 0])
        actual.fit(X_before, y_before)
        X_after = actual.transform(X_before).astype(float)

        X_test = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        assert np.allclose(X_after, X_test)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = OneHotEncoderComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        X_actual = actual.transform(X_test.copy())

        X_expected = pd.get_dummies(X_test, **config)

        assert np.allclose(X_actual, X_expected)
