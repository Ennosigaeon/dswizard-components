import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

from dswizard.components.data_preprocessing.imputation import ImputationComponent
from tests import base_test


class TestImputation(base_test.BaseComponentTest):

    def test_default(self):
        X_train = pd.DataFrame([[1.0], [np.nan]], columns=['foo'])
        y_train = pd.Series([1, 0])
        actual = ImputationComponent()
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_train)

        expected = SimpleImputer()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_train)

        assert actual.get_feature_names_out(['foo']).tolist() == ['foo']
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train = pd.DataFrame([[1.0], [np.nan], [5.0]], columns=['foo'])
        y_train = pd.Series([1, 0, 1])
        actual = ImputationComponent()
        config = self.get_config(actual)
        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_train)

        expected = SimpleImputer(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_train)

        if config['missing_indicator']:
            assert actual.get_feature_names_out(['foo']).tolist() == ['foo', 'missing_indicator']
        else:
            assert actual.get_feature_names_out(['foo']).tolist() == ['foo']
            assert np.allclose(X_actual, X_expected)

    def test_empty(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = ImputationComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test)

        expected = SimpleImputer()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
