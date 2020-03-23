import numpy as np
import sklearn

from automl.components.feature_preprocessing.truncated_svd import TruncatedSVDComponent
from tests import base_test


class TestTruncatedSVDComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = TruncatedSVDComponent(random_state=42)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.TruncatedSVD(random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = TruncatedSVDComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        # Fix n_components from percentage to absolute
        config['n_components'] = max(1, int(np.round(config['n_components'] * X_train.shape[1], 0)))

        expected = sklearn.decomposition.TruncatedSVD(**config, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)