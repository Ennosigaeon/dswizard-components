import numpy as np
from sklearn.feature_selection import VarianceThreshold

from dswizard.components.feature_preprocessing.variance_threshold import VarianceThresholdComponent
from tests import base_test


class TestVarianceThreshold(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = VarianceThresholdComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test)

        expected = VarianceThreshold(threshold=0.0001)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == feature_names
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = VarianceThresholdComponent()
        config: dict = self.get_config(actual, seed=0)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = VarianceThreshold(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == feature_names
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
