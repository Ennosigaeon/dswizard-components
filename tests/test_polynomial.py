import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from dswizard.components.feature_preprocessing.polynomial import PolynomialFeaturesComponent
from tests import base_test


class TestPolynomial(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = PolynomialFeaturesComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = PolynomialFeatures()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['1',
                                                                        'sepal length (cm)',
                                                                        'sepal width (cm)',
                                                                        'petal length (cm)',
                                                                        'petal width (cm)',
                                                                        'sepal length (cm)^2',
                                                                        'sepal length (cm) sepal width (cm)',
                                                                        'sepal length (cm) petal length (cm)',
                                                                        'sepal length (cm) petal width (cm)',
                                                                        'sepal width (cm)^2',
                                                                        'sepal width (cm) petal length (cm)',
                                                                        'sepal width (cm) petal width (cm)',
                                                                        'petal length (cm)^2',
                                                                        'petal length (cm) petal width (cm)',
                                                                        'petal width (cm)^2']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = PolynomialFeaturesComponent()
        config: dict = self.get_config(actual, seed=0)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = PolynomialFeatures(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['sepal length (cm)',
                                                                        'sepal width (cm)',
                                                                        'petal length (cm)',
                                                                        'petal width (cm)',
                                                                        'sepal length (cm)^2',
                                                                        'sepal length (cm) sepal width (cm)',
                                                                        'sepal length (cm) petal length (cm)',
                                                                        'sepal length (cm) petal width (cm)',
                                                                        'sepal width (cm)^2',
                                                                        'sepal width (cm) petal length (cm)',
                                                                        'sepal width (cm) petal width (cm)',
                                                                        'petal length (cm)^2',
                                                                        'petal length (cm) petal width (cm)',
                                                                        'petal width (cm)^2',
                                                                        'sepal length (cm)^3',
                                                                        'sepal length (cm)^2 sepal width (cm)',
                                                                        'sepal length (cm)^2 petal length (cm)',
                                                                        'sepal length (cm)^2 petal width (cm)',
                                                                        'sepal length (cm) sepal width (cm)^2',
                                                                        'sepal length (cm) sepal width (cm) petal length (cm)',
                                                                        'sepal length (cm) sepal width (cm) petal width (cm)',
                                                                        'sepal length (cm) petal length (cm)^2',
                                                                        'sepal length (cm) petal length (cm) petal width (cm)',
                                                                        'sepal length (cm) petal width (cm)^2',
                                                                        'sepal width (cm)^3',
                                                                        'sepal width (cm)^2 petal length (cm)',
                                                                        'sepal width (cm)^2 petal width (cm)',
                                                                        'sepal width (cm) petal length (cm)^2',
                                                                        'sepal width (cm) petal length (cm) petal width (cm)',
                                                                        'sepal width (cm) petal width (cm)^2',
                                                                        'petal length (cm)^3',
                                                                        'petal length (cm)^2 petal width (cm)',
                                                                        'petal length (cm) petal width (cm)^2',
                                                                        'petal width (cm)^3']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
