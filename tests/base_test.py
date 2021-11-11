from unittest import TestCase

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


class BaseComponentTest(TestCase):

    def test_default(self):
        pass

    def test_configured(self):
        pass

    def load_data(self, multiclass=True, categorical=False, random_state=42):
        np.random.seed(4)

        if categorical:
            X, y = datasets.fetch_openml(data_id=23381, return_X_y=True)
        elif multiclass:
            X, y = datasets.load_iris(return_X_y=True)
        else:
            X, y = datasets.load_breast_cancer(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def get_config(self, actual) -> dict:
        config: dict = actual.get_hyperparameter_search_space().sample_configuration().get_dictionary()
        print(config)
        return config

    def get_default(self, actual) -> dict:
        config: dict = actual.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
        print(config)
        return config
