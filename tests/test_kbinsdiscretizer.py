import numpy as np
import sklearn
from scipy.sparse import csr_matrix

from dswizard.components.feature_preprocessing.kbinsdiscretizer import KBinsDiscretizer
from tests import base_test


class TestKBinsDiscretizer(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = KBinsDiscretizer()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.KBinsDiscretizer()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)
        if isinstance(X_expected, csr_matrix):
            X_expected = X_expected.todense()

        assert actual.get_feature_names_out(feature_names).tolist() == ['sepal length (cm)_0.0',
                                                                        'sepal length (cm)_1.0',
                                                                        'sepal length (cm)_2.0',
                                                                        'sepal length (cm)_3.0',
                                                                        'sepal length (cm)_4.0', 'sepal width (cm)_0.0',
                                                                        'sepal width (cm)_1.0', 'sepal width (cm)_2.0',
                                                                        'sepal width (cm)_3.0', 'sepal width (cm)_4.0',
                                                                        'petal length (cm)_0.0',
                                                                        'petal length (cm)_1.0',
                                                                        'petal length (cm)_2.0',
                                                                        'petal length (cm)_3.0',
                                                                        'petal length (cm)_4.0', 'petal width (cm)_0.0',
                                                                        'petal width (cm)_1.0', 'petal width (cm)_2.0',
                                                                        'petal width (cm)_3.0', 'petal width (cm)_4.0']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = KBinsDiscretizer()
        config: dict = self.get_config(actual, seed=0)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.KBinsDiscretizer(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)
        if isinstance(X_expected, csr_matrix):
            X_expected = X_expected.todense()

        assert actual.get_feature_names_out(feature_names).tolist() == [
            'sepal length (cm)_0.0', 'sepal length (cm)_1.0', 'sepal length (cm)_2.0', 'sepal length (cm)_3.0',
            'sepal length (cm)_4.0', 'sepal length (cm)_5.0', 'sepal length (cm)_6.0', 'sepal length (cm)_7.0',
            'sepal length (cm)_8.0', 'sepal length (cm)_9.0', 'sepal length (cm)_10.0', 'sepal length (cm)_11.0',
            'sepal length (cm)_12.0', 'sepal length (cm)_13.0', 'sepal length (cm)_14.0', 'sepal length (cm)_15.0',
            'sepal length (cm)_16.0', 'sepal length (cm)_17.0', 'sepal length (cm)_18.0', 'sepal length (cm)_19.0',
            'sepal length (cm)_20.0', 'sepal length (cm)_21.0', 'sepal length (cm)_22.0', 'sepal length (cm)_23.0',
            'sepal length (cm)_24.0', 'sepal length (cm)_25.0', 'sepal length (cm)_26.0', 'sepal length (cm)_27.0',
            'sepal length (cm)_28.0', 'sepal length (cm)_29.0', 'sepal length (cm)_30.0', 'sepal length (cm)_31.0',
            'sepal length (cm)_32.0', 'sepal length (cm)_33.0', 'sepal length (cm)_34.0', 'sepal length (cm)_35.0',
            'sepal length (cm)_36.0', 'sepal length (cm)_37.0', 'sepal length (cm)_38.0', 'sepal length (cm)_39.0',
            'sepal length (cm)_40.0', 'sepal length (cm)_41.0', 'sepal length (cm)_42.0', 'sepal length (cm)_43.0',
            'sepal length (cm)_44.0', 'sepal length (cm)_45.0', 'sepal length (cm)_46.0', 'sepal length (cm)_47.0',
            'sepal length (cm)_48.0', 'sepal length (cm)_49.0', 'sepal length (cm)_50.0', 'sepal length (cm)_51.0',
            'sepal length (cm)_52.0', 'sepal length (cm)_53.0', 'sepal length (cm)_54.0', 'sepal length (cm)_55.0',
            'sepal length (cm)_56.0', 'sepal length (cm)_57.0', 'sepal length (cm)_58.0', 'sepal length (cm)_59.0',
            'sepal width (cm)_0.0', 'sepal width (cm)_1.0', 'sepal width (cm)_2.0', 'sepal width (cm)_3.0',
            'sepal width (cm)_4.0', 'sepal width (cm)_5.0', 'sepal width (cm)_6.0', 'sepal width (cm)_7.0',
            'sepal width (cm)_8.0', 'sepal width (cm)_9.0', 'sepal width (cm)_10.0', 'sepal width (cm)_11.0',
            'sepal width (cm)_12.0', 'sepal width (cm)_13.0', 'sepal width (cm)_14.0', 'sepal width (cm)_15.0',
            'sepal width (cm)_16.0', 'sepal width (cm)_17.0', 'sepal width (cm)_18.0', 'sepal width (cm)_19.0',
            'sepal width (cm)_20.0', 'sepal width (cm)_21.0', 'sepal width (cm)_22.0', 'sepal width (cm)_23.0',
            'sepal width (cm)_24.0', 'sepal width (cm)_25.0', 'sepal width (cm)_26.0', 'sepal width (cm)_27.0',
            'sepal width (cm)_28.0', 'sepal width (cm)_29.0', 'sepal width (cm)_30.0', 'sepal width (cm)_31.0',
            'sepal width (cm)_32.0', 'sepal width (cm)_33.0', 'sepal width (cm)_34.0', 'sepal width (cm)_35.0',
            'sepal width (cm)_36.0', 'sepal width (cm)_37.0', 'sepal width (cm)_38.0', 'sepal width (cm)_39.0',
            'sepal width (cm)_40.0', 'sepal width (cm)_41.0', 'sepal width (cm)_42.0', 'sepal width (cm)_43.0',
            'sepal width (cm)_44.0', 'sepal width (cm)_45.0', 'sepal width (cm)_46.0', 'sepal width (cm)_47.0',
            'sepal width (cm)_48.0', 'sepal width (cm)_49.0', 'sepal width (cm)_50.0', 'sepal width (cm)_51.0',
            'sepal width (cm)_52.0', 'sepal width (cm)_53.0', 'sepal width (cm)_54.0', 'sepal width (cm)_55.0',
            'sepal width (cm)_56.0', 'sepal width (cm)_57.0', 'sepal width (cm)_58.0', 'sepal width (cm)_59.0',
            'petal length (cm)_0.0', 'petal length (cm)_1.0', 'petal length (cm)_2.0', 'petal length (cm)_3.0',
            'petal length (cm)_4.0', 'petal length (cm)_5.0', 'petal length (cm)_6.0', 'petal length (cm)_7.0',
            'petal length (cm)_8.0', 'petal length (cm)_9.0', 'petal length (cm)_10.0', 'petal length (cm)_11.0',
            'petal length (cm)_12.0', 'petal length (cm)_13.0', 'petal length (cm)_14.0', 'petal length (cm)_15.0',
            'petal length (cm)_16.0', 'petal length (cm)_17.0', 'petal length (cm)_18.0', 'petal length (cm)_19.0',
            'petal length (cm)_20.0', 'petal length (cm)_21.0', 'petal length (cm)_22.0', 'petal length (cm)_23.0',
            'petal length (cm)_24.0', 'petal length (cm)_25.0', 'petal length (cm)_26.0', 'petal length (cm)_27.0',
            'petal length (cm)_28.0', 'petal length (cm)_29.0', 'petal length (cm)_30.0', 'petal length (cm)_31.0',
            'petal length (cm)_32.0', 'petal length (cm)_33.0', 'petal length (cm)_34.0', 'petal length (cm)_35.0',
            'petal length (cm)_36.0', 'petal length (cm)_37.0', 'petal length (cm)_38.0', 'petal length (cm)_39.0',
            'petal length (cm)_40.0', 'petal length (cm)_41.0', 'petal length (cm)_42.0', 'petal length (cm)_43.0',
            'petal length (cm)_44.0', 'petal length (cm)_45.0', 'petal length (cm)_46.0', 'petal length (cm)_47.0',
            'petal length (cm)_48.0', 'petal length (cm)_49.0', 'petal length (cm)_50.0', 'petal length (cm)_51.0',
            'petal length (cm)_52.0', 'petal length (cm)_53.0', 'petal length (cm)_54.0', 'petal length (cm)_55.0',
            'petal length (cm)_56.0', 'petal length (cm)_57.0', 'petal length (cm)_58.0', 'petal length (cm)_59.0',
            'petal width (cm)_0.0', 'petal width (cm)_1.0', 'petal width (cm)_2.0', 'petal width (cm)_3.0',
            'petal width (cm)_4.0', 'petal width (cm)_5.0', 'petal width (cm)_6.0', 'petal width (cm)_7.0',
            'petal width (cm)_8.0', 'petal width (cm)_9.0', 'petal width (cm)_10.0', 'petal width (cm)_11.0',
            'petal width (cm)_12.0', 'petal width (cm)_13.0', 'petal width (cm)_14.0', 'petal width (cm)_15.0',
            'petal width (cm)_16.0', 'petal width (cm)_17.0', 'petal width (cm)_18.0', 'petal width (cm)_19.0',
            'petal width (cm)_20.0', 'petal width (cm)_21.0', 'petal width (cm)_22.0', 'petal width (cm)_23.0',
            'petal width (cm)_24.0', 'petal width (cm)_25.0', 'petal width (cm)_26.0', 'petal width (cm)_27.0',
            'petal width (cm)_28.0', 'petal width (cm)_29.0', 'petal width (cm)_30.0', 'petal width (cm)_31.0',
            'petal width (cm)_32.0', 'petal width (cm)_33.0', 'petal width (cm)_34.0', 'petal width (cm)_35.0',
            'petal width (cm)_36.0', 'petal width (cm)_37.0', 'petal width (cm)_38.0', 'petal width (cm)_39.0',
            'petal width (cm)_40.0', 'petal width (cm)_41.0', 'petal width (cm)_42.0', 'petal width (cm)_43.0',
            'petal width (cm)_44.0', 'petal width (cm)_45.0', 'petal width (cm)_46.0', 'petal width (cm)_47.0',
            'petal width (cm)_48.0', 'petal width (cm)_49.0', 'petal width (cm)_50.0', 'petal width (cm)_51.0',
            'petal width (cm)_52.0', 'petal width (cm)_53.0', 'petal width (cm)_54.0', 'petal width (cm)_55.0',
            'petal width (cm)_56.0', 'petal width (cm)_57.0', 'petal width (cm)_58.0', 'petal width (cm)_59.0'
        ]
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
