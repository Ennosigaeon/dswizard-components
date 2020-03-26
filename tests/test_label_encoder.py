import numpy as np
import pandas as pd
import sklearn

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

    def test_categorical(self):

        actual = MultiColumnLabelEncoderComponent()
        X_before = pd.DataFrame([['Mann', 1], ['Frau', 2], ['Frau', 1]], dtype='category')
        y_before = pd.Series([1, 1, 0])
        actual.fit(X_before, y_before)
        X_after = actual.transform(X_before).astype(float)

        X_test = np.array([[1.0, 1], [0.0, 2], [0.0, 1]])

        assert np.allclose(X_after, X_test)

    def test_configured(self):
        pass
