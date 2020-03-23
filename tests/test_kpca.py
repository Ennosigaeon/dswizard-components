import numpy as np
import pytest
import sklearn

from automl.components.feature_preprocessing.kpca import KernelPCAComponent
from tests import base_test


class TestKernelPCAComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = KernelPCAComponent(random_state=42)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.KernelPCA(copy_X=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    @pytest.mark.skip
    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = KernelPCAComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.KernelPCA(**config, copy_X=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
