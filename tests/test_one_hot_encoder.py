import numpy as np
import pandas as pd

from dswizard.components.feature_preprocessing.one_hot_encoding import OneHotEncoderComponent
from tests import base_test


class TestOneHotEncoderComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = OneHotEncoderComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        df = pd.DataFrame(data=X_test, index=range(X_test.shape[0]), columns=range(X_test.shape[1]))
        X_expected = pd.get_dummies(df, sparse=False)

        assert np.allclose(X_actual, X_expected)

    def test_categorical(self):
        actual = OneHotEncoderComponent()
        X_before = pd.DataFrame([['Mann', 1], ['Frau', 2], ['Frau', 1]], columns=['Gender', 'Label'])
        y_before = pd.Series([1, 1, 0])
        X_after = actual.fit_transform(X_before.to_numpy(), y_before.to_numpy()).astype(float)

        X_expected = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 0.0, 1.0]])

        assert np.allclose(X_after, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = OneHotEncoderComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        X_actual = actual.fit_transform(X_test.copy())

        df = pd.DataFrame(data=X_test, index=range(X_test.shape[0]), columns=range(X_test.shape[1]))
        X_expected = pd.get_dummies(df, **config)

        assert np.allclose(X_actual, X_expected)
