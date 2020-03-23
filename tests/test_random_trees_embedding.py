import numpy as np
import sklearn

from automl.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbeddingComponent
from tests import base_test


class TestRandomTreesEmbeddingComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RandomTreesEmbeddingComponent(random_state=42)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.ensemble.RandomTreesEmbedding(random_state=42, sparse_output=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RandomTreesEmbeddingComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.ensemble.RandomTreesEmbedding(**config, sparse_output=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.preprocessor) == repr(expected)
        assert np.allclose(X_actual, X_expected)
