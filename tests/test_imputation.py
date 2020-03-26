import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

from automl.components.data_preprocessing.imputation import ImputationComponent
from tests import base_test


class TestImputation(base_test.BaseComponentTest):

    def test_default(self):
        X_train = pd.DataFrame([[1.0], [np.nan]])
        y_train = pd.Series([1, 0])
        actual = ImputationComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_train)

        expected = SimpleImputer(copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_train)

        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train = pd.DataFrame([[1.0], [np.nan], [5.0]])
        y_train = pd.Series([1, 0, 1])
        actual = ImputationComponent()
        config: dict = self.get_config(actual)
        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_train)

        expected = SimpleImputer(**config, copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_train)

        assert np.allclose(X_actual, X_expected)

    def test_empty(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = ImputationComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test)

        expected = SimpleImputer(copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
