import numpy as np
import pandas as pd

from dswizard.components.feature_preprocessing.one_hot_encoding import OneHotEncoderComponent
from tests import base_test


class TestOneHotEncoderComponent(base_test.BaseComponentTest):

    def test_default_numerical(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        actual = OneHotEncoderComponent()
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        df = pd.DataFrame(data=X_test, index=range(X_test.shape[0]), columns=range(X_test.shape[1]))
        X_expected = pd.get_dummies(df, sparse=False)

        assert actual.get_feature_names_out(feature_names).tolist() == feature_names
        assert np.allclose(X_actual, X_expected)

    # noinspection PyUnusedLocal
    def test_default_categorical(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data(categorical=True)

        actual = OneHotEncoderComponent()
        config = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test.copy())

        df = pd.DataFrame(data=X_test, index=range(X_test.shape[0]), columns=range(X_test.shape[1]))
        X_expected = pd.get_dummies(df, sparse=False)

        assert set(actual.get_feature_names_out(feature_names).tolist()) == {
            'V2_Brief', 'V2_Casual', 'V2_Flare', 'V2_Novelty', 'V2_OL', 'V2_Sexy', 'V2_bohemian', 'V2_cute',
            'V2_fashion', 'V2_party', 'V2_sexy', 'V2_vintage', 'V2_work', 'V3_Average', 'V3_High', 'V3_Low',
            'V3_Medium', 'V3_high', 'V3_low', 'V3_very-high', 'V3_nan', 'V5_L', 'V5_M', 'V5_S', 'V5_XL', 'V5_free',
            'V5_s', 'V5_small', 'V6_Automn', 'V6_Autumn', 'V6_Spring', 'V6_Summer', 'V6_Winter', 'V6_summer',
            'V6_winter', 'V6_nan', 'V7_Scoop', 'V7_Sweetheart', 'V7_backless', 'V7_boat-neck', 'V7_bowneck',
            'V7_halter', 'V7_mandarin-collor', 'V7_o-neck', 'V7_open', 'V7_peterpan-collor', 'V7_slash-neck',
            'V7_sqare-collor', 'V7_turndowncollor', 'V7_v-neck', 'V7_nan', 'V8_Petal', 'V8_butterfly', 'V8_cap-sleeves',
            'V8_capsleeves', 'V8_full', 'V8_half', 'V8_halfsleeve', 'V8_short', 'V8_sleeevless', 'V8_sleeveless',
            'V8_sleevless', 'V8_sleveless', 'V8_threequarter', 'V8_threequater', 'V8_thressqatar', 'V8_turndowncollor',
            'V8_nan', 'V9_dropped', 'V9_empire', 'V9_natural', 'V9_princess', 'V9_nan', 'V10_acrylic', 'V10_cashmere',
            'V10_chiffonfabric', 'V10_cotton', 'V10_knitting', 'V10_lace', 'V10_linen', 'V10_lycra', 'V10_microfiber',
            'V10_milksilk', 'V10_mix', 'V10_modal', 'V10_nylon', 'V10_other', 'V10_polyster', 'V10_rayon',
            'V10_shiffon', 'V10_silk', 'V10_sill', 'V10_spandex', 'V10_nan', 'V11_Corduroy', 'V11_broadcloth',
            'V11_chiffon', 'V11_dobby', 'V11_flannel', 'V11_jersey', 'V11_knitted', 'V11_knitting', 'V11_lace',
            'V11_organza', 'V11_poplin', 'V11_satin', 'V11_sattin', 'V11_shiffon', 'V11_tulle', 'V11_wollen',
            'V11_woolen', 'V11_worsted', 'V11_nan', 'V12_applique', 'V12_beading', 'V12_bow', 'V12_button',
            'V12_cascading', 'V12_crystal', 'V12_draped', 'V12_embroidary', 'V12_feathers', 'V12_flowers',
            'V12_hollowout', 'V12_lace', 'V12_none', 'V12_pearls', 'V12_pleat', 'V12_pockets', 'V12_rivet',
            'V12_ruched', 'V12_ruffles', 'V12_sashes', 'V12_sequined', 'V12_tassel', 'V12_nan', 'V13_animal',
            'V13_character', 'V13_dot', 'V13_floral', 'V13_geometric', 'V13_leapord', 'V13_leopard', 'V13_patchwork',
            'V13_plaid', 'V13_print', 'V13_solid', 'V13_striped', 'V13_nan', 'V4'}
        # TODO: pd.get_dummies does not exactly behave like sklearn OHE
        # assert np.allclose(X_actual, X_expected)

    def test_categorical(self):
        actual = OneHotEncoderComponent()
        X_before = pd.DataFrame([['Mann', 1], ['Frau', 2], ['Frau', 1]], columns=['Gender', 'Label'])
        y_before = pd.Series([1, 1, 0])
        X_after = actual.fit_transform(X_before.to_numpy(), y_before.to_numpy()).astype(float)

        X_expected = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 0.0, 1.0]])

        assert np.allclose(X_after, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test, feature_names = self.load_data(categorical=True)

        actual = OneHotEncoderComponent()
        config = self.get_config(actual)

        actual.set_hyperparameters(config)
        X_actual = actual.fit_transform(X_test.copy())

        df = pd.DataFrame(data=X_test, index=range(X_test.shape[0]), columns=range(X_test.shape[1]))
        X_expected = pd.get_dummies(df, **config)

        assert set(actual.get_feature_names_out(feature_names).tolist()) == {
            'V2_Brief', 'V2_Casual', 'V2_Flare', 'V2_Novelty', 'V2_Sexy', 'V2_bohemian', 'V2_cute', 'V2_party',
            'V2_sexy', 'V2_vintage', 'V2_work', 'V3_Average', 'V3_High', 'V3_Low', 'V3_Medium', 'V3_high', 'V3_low',
            'V3_very-high', 'V3_nan', 'V5_L', 'V5_M', 'V5_S', 'V5_XL', 'V5_free', 'V6_Automn', 'V6_Autumn', 'V6_Spring',
            'V6_Summer', 'V6_Winter', 'V6_spring', 'V6_winter', 'V7_Scoop', 'V7_Sweetheart', 'V7_boat-neck',
            'V7_bowneck', 'V7_o-neck', 'V7_peterpan-collor', 'V7_ruffled', 'V7_slash-neck', 'V7_sqare-collor',
            'V7_sweetheart', 'V7_turndowncollor', 'V7_v-neck', 'V7_nan', 'V8_cap-sleeves', 'V8_full', 'V8_halfsleeve',
            'V8_short', 'V8_sleeveless', 'V8_sleevless', 'V8_threequarter', 'V8_thressqatar', 'V8_urndowncollor',
            'V8_nan', 'V9_dropped', 'V9_empire', 'V9_natural', 'V9_nan', 'V10_acrylic', 'V10_cashmere',
            'V10_chiffonfabric', 'V10_cotton', 'V10_lycra', 'V10_milksilk', 'V10_mix', 'V10_model', 'V10_nylon',
            'V10_other', 'V10_polyster', 'V10_rayon', 'V10_shiffon', 'V10_silk', 'V10_spandex', 'V10_viscos',
            'V10_wool', 'V10_nan', 'V11_batik', 'V11_broadcloth', 'V11_chiffon', 'V11_dobby', 'V11_flannael',
            'V11_jersey', 'V11_other', 'V11_poplin', 'V11_sattin', 'V11_shiffon', 'V11_terry', 'V11_wollen',
            'V11_worsted', 'V11_nan', 'V12_Tiered', 'V12_applique', 'V12_beading', 'V12_bow', 'V12_button',
            'V12_crystal', 'V12_embroidary', 'V12_feathers', 'V12_flowers', 'V12_hollowout', 'V12_lace', 'V12_plain',
            'V12_pockets', 'V12_rivet', 'V12_ruched', 'V12_ruffles', 'V12_sashes', 'V12_sequined', 'V12_nan',
            'V13_animal', 'V13_dot', 'V13_floral', 'V13_geometric', 'V13_none', 'V13_patchwork', 'V13_print',
            'V13_solid', 'V13_splice', 'V13_striped', 'V13_nan', 'V4'}
        # TODO: pd.get_dummies does not exactly behave like sklearn OHE
        # assert np.allclose(X_actual, X_expected)
