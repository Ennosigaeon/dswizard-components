import numpy as np

from dswizard.components.feature_preprocessing.missing_indicator import MissingIndicatorComponent
from tests import base_test


class TestMissingIndicatorComponent(base_test.BaseComponentTest):

    def test_missing(self):
        X = np.array([[np.nan, 1, 3],
                      [4, 0, np.nan],
                      [8, 1, 0],
                      [8, 1, 0],
                      [4, 0, np.nan]])
        feature_names = ['1', '2', '3']

        actual = MissingIndicatorComponent()
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X)
        X_actual = actual.transform(np.copy(X))

        assert actual.get_feature_names_out(feature_names).tolist() == ['missing_values']
        assert np.allclose(X_actual, np.array([[1, 1, 0, 0, 1]]).T)

    def test_no_missing(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = MissingIndicatorComponent()
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train)
        X_actual = actual.transform(np.copy(X_train))

        assert actual.get_feature_names_out(feature_names).tolist() == ['missing_values']
        assert np.allclose(X_actual, np.zeros((100, 1)))
