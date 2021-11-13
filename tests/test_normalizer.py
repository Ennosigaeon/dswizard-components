import numpy as np
from sklearn.preprocessing import Normalizer

from dswizard.components.data_preprocessing.normalize import NormalizerComponent
from tests import base_test


class TestNormalizer(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = NormalizerComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = Normalizer()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == feature_names
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = NormalizerComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = Normalizer(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == feature_names
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
