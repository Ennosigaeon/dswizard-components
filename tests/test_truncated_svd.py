import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.truncated_svd import TruncatedSVDComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestTruncatedSVDComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = TruncatedSVDComponent(random_state=42)
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.TruncatedSVD(random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['singular_value_0', 'singular_value_1']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = TruncatedSVDComponent(random_state=42)
        config = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        # Fix n_components from percentage to absolute
        config['n_components'] = min(resolve_factor(config['n_components_factor'], X_train.shape[1]),
                                     (X_train.shape[1] - 1))
        del config['n_components_factor']

        expected = sklearn.decomposition.TruncatedSVD(**config, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['singular_value_{}'.format(i) for i in
                                                                        range(expected.explained_variance_.shape[0])]
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
