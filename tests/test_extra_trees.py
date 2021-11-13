import numpy as np
import sklearn.tree

from dswizard.components.classification.excluded.extra_trees import ExtraTreesClassifier
from dswizard.components.util import resolve_factor
from tests import base_test


class TestExtraTreesClassifier(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = ExtraTreesClassifier(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.ensemble.ExtraTreesClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['prediction']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = ExtraTreesClassifier(random_state=42)
        config = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['max_depth'] = max(resolve_factor(config['max_depth_factor'], X_train.shape[1]), 2)
        del config['max_depth_factor']

        config['max_leaf_nodes'] = max(resolve_factor(config['max_leaf_nodes_factor'], X_train.shape[0]), 2)
        del config['max_leaf_nodes_factor']

        expected = sklearn.ensemble.ExtraTreesClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['prediction']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(y_actual, y_expected)
