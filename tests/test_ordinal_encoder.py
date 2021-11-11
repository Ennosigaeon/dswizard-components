import numpy as np
import pandas as pd

from dswizard.components.feature_preprocessing.ordinal_encoder import OrdinalEncoderComponent
from tests import base_test


class TestLabelEncoderComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = OrdinalEncoderComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        assert np.allclose(X_actual, X_test)

    def test_categorical(self):
        actual = OrdinalEncoderComponent()
        X_before = pd.DataFrame([['Mann', np.nan], ['Frau', 2], [None, 1]], columns=['Gender', 'Label'],
                                dtype='category').to_numpy()
        y_before = pd.Series([1, 1, 0]).to_numpy()
        actual.fit(X_before, y_before)
        X_after = actual.transform(X_before).astype(float)

        X_test = np.array([[2.0, np.nan], [1.0, 1.0], [np.nan, 0.0]])

        assert np.allclose(X_after, X_test, equal_nan=True)

    def test_configured(self):
        pass
