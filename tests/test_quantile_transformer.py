import numpy as np
import sklearn

from automl.components.data_preprocessing.test.quantile_transformer import QuantileTransformerComponent
from tests import base_test


class TestQuantileTransformerComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = QuantileTransformerComponent(random_state=42)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.QuantileTransformer(random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = QuantileTransformerComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.QuantileTransformer(**config,random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)