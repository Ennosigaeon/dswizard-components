import numpy as np
import pytest
import sklearn

from automl.components.feature_preprocessing.one_hot_encoding import OneHotEncoderComponent
from tests import base_test


class TestOneHotEncoderComponent(base_test.BaseComponentTest):

    # TODO test for categorical data is missing

    @pytest.mark.skip
    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = OneHotEncoderComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        expected = sklearn.preprocessing.OneHotEncoder(sparse=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    @pytest.mark.skip
    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = OneHotEncoderComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        expected = sklearn.preprocessing.OneHotEncoder(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
