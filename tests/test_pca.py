import numpy as np
from sklearn.decomposition import PCA

from dswizard.components.feature_preprocessing.pca import PCAComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestPCA(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = PCAComponent(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(np.copy(X_train), np.copy(y_train))
        X_actual = actual.transform(np.copy(X_test))

        expected = PCA(copy=False, random_state=42)
        expected.fit(np.copy(X_train), np.copy(y_train))
        X_expected = expected.transform(np.copy(X_test))

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = PCAComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(np.copy(X_train), np.copy(y_train))
        X_actual = actual.transform(np.copy(X_test))

        config['n_components'] = resolve_factor(config['keep_variance'], min(*X_train.shape))
        del config['keep_variance']

        expected = PCA(**config, copy=False, random_state=42)
        expected.fit(np.copy(X_train), np.copy(y_train))
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
