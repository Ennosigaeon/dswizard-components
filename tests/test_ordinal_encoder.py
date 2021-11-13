import numpy as np
import pandas as pd

from dswizard.components.feature_preprocessing.ordinal_encoder import OrdinalEncoderComponent
from tests import base_test


class TestOrdinalEncoderComponent(base_test.BaseComponentTest):

    def test_default_numerical(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = OrdinalEncoderComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        assert actual.get_feature_names_out(feature_names).tolist() == feature_names
        assert np.allclose(X_actual, X_test)

    def test_default_categorical(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data(categorical=True)

        actual = OrdinalEncoderComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        assert actual.get_feature_names_out(feature_names).tolist() == ['V2', 'V3', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                                                        'V11', 'V12', 'V13', 'V4']
        # assert np.allclose(X_actual, X_test)

    def test_categorical(self):
        actual = OrdinalEncoderComponent()
        X = pd.DataFrame([['Mann', np.nan], ['Frau', 2], [None, 1]], columns=['Gender', 'Label'],
                                dtype='category').to_numpy()
        y_before = pd.Series([1, 1, 0]).to_numpy()
        actual.fit(X, y_before)
        X_actual = actual.transform(X).astype(float)

        X_expected = np.array([[1.0, np.nan], [0.0, 2.0], [np.nan, 1.0]])

        assert actual.get_feature_names_out(['Gender', 'Label']).tolist() == ['Gender', 'Label']
        assert np.allclose(X_actual, X_expected, equal_nan=True)

    def test_configured(self):
        pass
