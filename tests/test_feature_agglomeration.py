import numpy as np
import sklearn

from automl.components.feature_preprocessing.feature_agglomeration import FeatureAgglomerationComponent
from automl.util.common import resolve_factor
from tests import base_test


class TestFeatureAgglomerationComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = FeatureAgglomerationComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.cluster.FeatureAgglomeration()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = FeatureAgglomerationComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        if config['pooling_func'] is "mean":
            config['pooling_func'] = np.mean
        elif config['pooling_func'] is "median":
            config['pooling_func'] = np.median
        elif config['pooling_func'] is "max":
            config['pooling_func'] = np.max

        config['n_clusters'] = resolve_factor(config['n_clusters_factor'], X_train.shape[1])
        del config['n_clusters_factor']

        expected = sklearn.cluster.FeatureAgglomeration(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
