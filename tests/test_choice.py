import base_test
from components.classification import ClassifierChoice
from components.data_preprocessing import DataPreprocessorChoice
from components.feature_preprocessing import FeaturePreprocessorChoice


class ChoiceTest(base_test.BaseComponentTest):

    def test_classifier(self):
        component = ClassifierChoice()
        component.get_components()
        component.get_hyperparameter_search_space()

    def test_data(self):
        component = DataPreprocessorChoice()
        component.get_components()
        component.get_hyperparameter_search_space()

    def test_feature(self):
        component = FeaturePreprocessorChoice()
        component.get_components()
        component.get_hyperparameter_search_space()
