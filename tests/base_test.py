from unittest import TestCase

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


class BaseComponentTest(TestCase):

    def test_default(self):
        pass

    def test_configured(self):
        pass

    def load_data(self, multiclass=True, categorical=False, random_state=42):
        def _to_pd(array):
            return pd.DataFrame(data=array, index=range(array.shape[0]), columns=range(array.shape[1]))

        def _to_series(array):
            return pd.Series(array, index=range(array.shape[0]))

        if categorical:
            X, y = datasets.fetch_openml(data_id=23381, return_X_y=True)
        elif multiclass:
            X, y = datasets.load_iris(True)
        else:
            X, y = datasets.load_breast_cancer(True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
        return _to_pd(X_train), _to_pd(X_test), _to_series(y_train), _to_series(y_test)

    def get_config(self, actual) -> dict:
        config: dict = actual.get_hyperparameter_search_space().sample_configuration().get_dictionary()
        # for key, value in config.items():
        #     if check_none(value):
        #         config[key] = None
        print(config)
        return config
