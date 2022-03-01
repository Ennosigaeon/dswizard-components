import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.fast_ica import FastICAComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestFastICAComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = FastICAComponent(random_state=42)
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.FastICA(random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['independent_component_{}'.format(i) for i in
                                                                        range(4)]
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = FastICAComponent(random_state=42)
        config = self.get_config(actual, seed=0)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        config['n_components'] = resolve_factor(config['n_components_factor'], min(*X_train.shape))
        del config['n_components_factor']

        expected = sklearn.decomposition.FastICA(**config, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['independent_component_{}'.format(i) for i in
                                                                        range(4)]
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
