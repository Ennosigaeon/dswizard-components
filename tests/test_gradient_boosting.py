import numpy as np
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

from automl.components.classification.gradient_boosting import GradientBoostingClassifier
from automl.util.common import resolve_factor
from tests import base_test


class TestGradientBoosting(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GradientBoostingClassifier(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = HistGradientBoostingClassifier(n_iter_no_change=10, l2_regularization=1e-10, max_iter=512,
                                                  random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data(multiclass=False)

        actual = GradientBoostingClassifier(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['max_depth'] = max(resolve_factor(config['max_depth_factor'], X_train.shape[1]), 2)
        del config['max_depth_factor']

        config['max_leaf_nodes'] = max(resolve_factor(config['max_leaf_nodes_factor'], X_train.shape[0]), 2)
        del config['max_leaf_nodes_factor']

        config['min_samples_leaf'] = resolve_factor(config['min_samples_leaf_factor'], X_train.shape[0])
        del config['min_samples_leaf_factor']

        expected = HistGradientBoostingClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)
