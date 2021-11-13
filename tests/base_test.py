from unittest import TestCase

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class BaseComponentTest(TestCase):

    def test_default(self):
        pass

    def test_configured(self):
        pass

    def load_data(self, multiclass=True, categorical=False, random_state=42):
        np.random.seed(4)

        if categorical:
            X, y = datasets.fetch_openml(data_id=23381, as_frame=True, return_X_y=True)
        elif multiclass:
            X, y = datasets.load_iris(as_frame=True, return_X_y=True)
        else:
            X, y = datasets.load_breast_cancer(as_frame=True, return_X_y=True)

        feature_names = X.columns.to_list()

        X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.33,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test, feature_names

    def get_config(self, actual, seed: int = None) -> dict:
        cs = actual.get_hyperparameter_search_space()
        if seed is not None:
            cs.seed(seed)
        config: dict = cs.sample_configuration().get_dictionary()
        print(config)
        return config

    def get_default(self, actual) -> dict:
        config: dict = actual.get_hyperparameter_search_space().get_default_configuration().get_dictionary()
        print(config)
        return config
