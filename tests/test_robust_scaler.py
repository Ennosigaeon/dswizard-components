import numpy as np
from sklearn.preprocessing import RobustScaler

from automl.components.data_preprocessing.robust_scaler import RobustScalerComponent
from tests import base_test


class TestRobustScaler(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RobustScalerComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = RobustScaler(copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RobustScalerComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        if config['q_max'] < config['q_min']:
            help = config['q_max']
            config['q_max'] = config['q_min']
            config['q_min'] = help

        expected = RobustScaler(quantile_range=(config['q_min'], config['q_max']),
                                with_centering=config['with_centering'], with_scaling=config['with_scaling'],
                                copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
