import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.select_percentile import SelectPercentileClassification
from tests import base_test


class TestSelectPercentileClassification(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = SelectPercentileClassification()
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.feature_selection.SelectPercentile()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['petal length (cm)']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = SelectPercentileClassification()
        config = self.get_config(actual, seed=0)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        if config['score_func'] == "chi2":
            config['score_func'] = sklearn.feature_selection.chi2
        elif config['score_func'] == "f_classif":
            config['score_func'] = sklearn.feature_selection.f_classif
        elif config['score_func'] == "mutual_info":
            config['score_func'] = sklearn.feature_selection.mutual_info_classif

        expected = sklearn.feature_selection.SelectPercentile(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['petal length (cm)', 'petal width (cm)']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
