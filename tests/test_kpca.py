import numpy as np
import pytest
import sklearn

from automl.components.feature_preprocessing.kpca import KernelPCAComponent
from automl.util.common import resolve_factor
from tests import base_test


class TestKernelPCAComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = KernelPCAComponent(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.KernelPCA(kernel='rbf', n_jobs=1, copy_X=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()
        np.random.seed(0)

        actual = KernelPCAComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        config['n_components'] = resolve_factor(config['n_components_factor'], min(*X_train.shape))
        del config['n_components_factor']

        expected = sklearn.decomposition.KernelPCA(**config, n_jobs=1, copy_X=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
