import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.impute import MissingIndicator

from dswizard.components.base import PreprocessingAlgorithm, NoopComponent
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS


class ImputationComponent(PreprocessingAlgorithm):
    """ImputationComponent

    An ImputationComponent that can handle missing values. Missing values are automatically detected
    Read more in the :ref:`User Guide`.

    Parameters
    ----------
    missing_values : np.nan
        The value that is treated as a missing value

    strategy : str
        The imputation strategy

    add_indicator : bool
        Add an indicator if a row contained a missing value

    Attributes
    ----------

    See also
    --------
    SimpleImputer

    References
    ----------
    """

    def __init__(self, missing_values=np.nan, strategy: str = 'mean', add_indicator: bool = False):
        super().__init__()
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.missing_values = missing_values

    def fit(self, X, y=None):
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1]))

        if not np.any(pd.isna(X)):
            self.preprocessor = NoopComponent()
            return self
        else:
            numeric = X.select_dtypes(include=['number']).columns
            categorical = X.select_dtypes(include=['category', 'object']).columns

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', SimpleImputer(missing_values=self.missing_values, strategy='most_frequent',
                                      add_indicator=False, copy=False), categorical),
                ('num', SimpleImputer(missing_values=self.missing_values, strategy=self.strategy,
                                      add_indicator=False, copy=False), numeric)
            ]
        )
        self.preprocessor.fit(X)

        return self

    def transform(self, X):

        if self.add_indicator:
            missingIndicator = MissingIndicator()
            X_missing = missingIndicator.fit_transform(X)
            X_missing = pd.DataFrame(X_missing)
            newdf = pd.DataFrame()
            for index, row in X_missing.iterrows():
                if row.any():
                    newdf = newdf.append({'missing': True}, ignore_index=True)
                else:
                    newdf = newdf.append({'missing': False}, ignore_index=True)

        if self.preprocessor is None:
            raise NotImplementedError()
        X_new = self.preprocessor.transform(X)

        if self.add_indicator:
            X_new = pd.concat([pd.DataFrame(X_new), newdf], axis=1, sort=False).to_numpy()
        return X_new

    @staticmethod
    def get_properties():
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter("strategy", ["mean", "median", "most_frequent"], default_value="mean")
        add_indicator = CategoricalHyperparameter("add_indicator", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([strategy, add_indicator])
        return cs
