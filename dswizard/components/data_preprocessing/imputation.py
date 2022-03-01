from typing import List

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.pipeline import FeatureUnion

from dswizard.components.base import PreprocessingAlgorithm
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
        super().__init__('imputation')
        try:
            if np.isnan(missing_values):
                self.args['missing_values'] = 'NaN'
        except TypeError:
            pass

        self.strategy = strategy
        self.add_indicator = add_indicator
        self.missing_values = missing_values

    def fit(self, X, y=None):
        from dswizard.components.feature_preprocessing.missing_indicator import MissingIndicatorComponent
        from dswizard.components.base import NoopComponent
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer

        df = pd.DataFrame(data=X, index=range(X.shape[0]), columns=range(X.shape[1]))

        if not np.any(pd.isna(df)):
            self.estimator_ = NoopComponent()
            return self
        else:
            numeric = df.select_dtypes(include=['number']).columns
            categorical = df.select_dtypes(include=['category', 'object']).columns

        imputer = ColumnTransformer(
            transformers=[
                ('cat', SimpleImputer(missing_values=self.missing_values, strategy='most_frequent',
                                      add_indicator=False), categorical),
                ('num', SimpleImputer(missing_values=self.missing_values, strategy=self.strategy,
                                      add_indicator=False), numeric)
            ]
        )
        if self.add_indicator:
            self.estimator_ = FeatureUnion(transformer_list=[
                ('imputation', imputer), ('indicator', MissingIndicatorComponent())])
        else:
            self.estimator_ = imputer
        self.estimator_.fit(df)

        return self

    def get_feature_names_out(self, input_features: List[str] = None):
        from sklearn.utils.validation import _check_feature_names_in

        names = _check_feature_names_in(self, input_features)
        if self.add_indicator:
            imp = self.estimator_.transformer_list[0][1]
            mis = self.estimator_.transformer_list[1][1]
            names = np.append(names[imp._columns[0]], names[imp._columns[1]])
            names = np.append(names, mis.get_feature_names_out())
        else:
            names = np.append(names[self.estimator_._columns[0]], names[self.estimator_._columns[1]])

        return names

    @staticmethod
    def get_properties():
        return {'shortname': 'Simple Imputation',
                'name': 'Simple Imputation',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def deserialize(**kwargs) -> 'ImputationComponent':
        if 'missing_values' in kwargs and kwargs['missing_values'] == 'NaN':
            kwargs['missing_values'] = np.nan
        return ImputationComponent(**kwargs)

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter("strategy", ["mean", "median", "most_frequent"], default_value="mean")
        add_indicator = CategoricalHyperparameter("add_indicator", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([strategy, add_indicator])
        return cs
