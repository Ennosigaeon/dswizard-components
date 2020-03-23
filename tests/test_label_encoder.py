import numpy as np

from automl.components.feature_preprocessing.multi_column_label_encoder import MultiColumnLabelEncoderComponent
from tests import base_test


class TestLabelEncoderComponent(base_test.BaseComponentTest):

    # TODO add test case with categorical and numerical features

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = MultiColumnLabelEncoderComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        assert np.allclose(X_actual, X_test)

    def test_configured(self):
        pass
