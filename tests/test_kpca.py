import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.kpca import KernelPCAComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestKernelPCAComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = KernelPCAComponent(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.KernelPCA(kernel='rbf', n_jobs=1, copy_X=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == [
            'principal_component_0', 'principal_component_1', 'principal_component_2', 'principal_component_3',
            'principal_component_4', 'principal_component_5', 'principal_component_6', 'principal_component_7',
            'principal_component_8', 'principal_component_9', 'principal_component_10', 'principal_component_11',
            'principal_component_12', 'principal_component_13', 'principal_component_14', 'principal_component_15',
            'principal_component_16', 'principal_component_17', 'principal_component_18', 'principal_component_19',
            'principal_component_20', 'principal_component_21', 'principal_component_22', 'principal_component_23',
            'principal_component_24', 'principal_component_25', 'principal_component_26', 'principal_component_27',
            'principal_component_28', 'principal_component_29', 'principal_component_30', 'principal_component_31',
            'principal_component_32', 'principal_component_33', 'principal_component_34', 'principal_component_35',
            'principal_component_36', 'principal_component_37', 'principal_component_38', 'principal_component_39',
            'principal_component_40', 'principal_component_41', 'principal_component_42', 'principal_component_43',
            'principal_component_44', 'principal_component_45', 'principal_component_46', 'principal_component_47',
            'principal_component_48', 'principal_component_49', 'principal_component_50', 'principal_component_51',
            'principal_component_52', 'principal_component_53', 'principal_component_54', 'principal_component_55',
            'principal_component_56', 'principal_component_57', 'principal_component_58', 'principal_component_59',
            'principal_component_60', 'principal_component_61', 'principal_component_62', 'principal_component_63',
            'principal_component_64', 'principal_component_65', 'principal_component_66', 'principal_component_67',
            'principal_component_68', 'principal_component_69', 'principal_component_70', 'principal_component_71',
            'principal_component_72', 'principal_component_73', 'principal_component_74', 'principal_component_75',
            'principal_component_76', 'principal_component_77', 'principal_component_78', 'principal_component_79',
            'principal_component_80', 'principal_component_81', 'principal_component_82', 'principal_component_83',
            'principal_component_84', 'principal_component_85', 'principal_component_86', 'principal_component_87',
            'principal_component_88', 'principal_component_89', 'principal_component_90', 'principal_component_91',
            'principal_component_92', 'principal_component_93', 'principal_component_94', 'principal_component_95',
            'principal_component_96', 'principal_component_97', 'principal_component_98']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = KernelPCAComponent(random_state=42)
        config: dict = self.get_config(actual, seed=0)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        config['n_components'] = resolve_factor(config['n_components_factor'], min(*X_train.shape))
        del config['n_components_factor']

        expected = sklearn.decomposition.KernelPCA(**config, n_jobs=1, copy_X=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert actual.get_feature_names_out(feature_names).tolist() == ['principal_component_0',
                                                                        'principal_component_1']
        assert repr(actual.estimator_) == repr(expected)
        assert np.allclose(X_actual, X_expected)
