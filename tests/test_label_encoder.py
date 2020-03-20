import numpy as np
import sklearn

from automl.components.feature_preprocessing.test.multi_column_label_encoder import MultiColumnLabelEncoderComponent
from tests import base_test


class TestLabelEncoderComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = MultiColumnLabelEncoderComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.LabelEncoder()
        expected.fit(X_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = MultiColumnLabelEncoderComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.LabelEncoder()
        expected.fit(X_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)
