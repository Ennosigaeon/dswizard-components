import numpy as np
import pytest
import sklearn.naive_bayes
import sklearn.svm

from automl.components.classification.gaussian_process import GaussianProcessClassifier
from tests import base_test


class TestGaussianProcessClassifier(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GaussianProcessClassifier(random_state=42)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.gaussian_process.GaussianProcessClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    @pytest.mark.skip
    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GaussianProcessClassifier(random_state=42)
        config: dict = self.get_config(actual)

        if config['kernel'] is "constant":
            from sklearn.gaussian_process.kernels import ConstantKernel
            config['kernel'] = ConstantKernel()
        elif config['kernel'] is "rbf":
            from sklearn.gaussian_process.kernels import RBF
            config['kernel'] = RBF()
        elif config['kernel'] is "matern":
            from sklearn.gaussian_process.kernels import Matern
            config['kernel'] = Matern()
        elif config['kernel'] is "rational_quadratic":
            from sklearn.gaussian_process.kernels import RationalQuadratic
            config['kernel'] = RationalQuadratic()
        elif config['kernel'] is "exp_sin_squared":
            from sklearn.gaussian_process.kernels import ExpSineSquared
            config['kernel'] = ExpSineSquared()
        elif config['kernel'] is "white":
            from sklearn.gaussian_process.kernels import WhiteKernel
            config['kernel'] = WhiteKernel()
        elif config['kernel'] is "dot":
            from sklearn.gaussian_process.kernels import DotProduct
            config['kernel'] = DotProduct()

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.gaussian_process.GaussianProcessClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)
