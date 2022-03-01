import base_test
from dswizard.components.classification import ClassifierChoice
from dswizard.components.data_preprocessing import DataPreprocessorChoice
from dswizard.components.feature_preprocessing import FeaturePreprocessorChoice


class ChoiceTest(base_test.BaseComponentTest):

    @staticmethod
    def test_classifier():
        component = ClassifierChoice()
        component.get_components()
        component.get_hyperparameter_search_space()

    @staticmethod
    def test_data():
        component = DataPreprocessorChoice()
        component.get_components()
        component.get_hyperparameter_search_space()

    @staticmethod
    def test_feature():
        component = FeaturePreprocessorChoice()
        component.get_components()
        component.get_hyperparameter_search_space()
