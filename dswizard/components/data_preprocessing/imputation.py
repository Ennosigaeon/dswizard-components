import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.pipeline import FeatureUnion

from dswizard.components.base import PreprocessingAlgorithm, NoopComponent
from dswizard.components.feature_preprocessing.missing_indicator import MissingIndicatorComponent
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
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.missing_values = missing_values

    def fit(self, X, y=None):
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

    def get_feature_names_out(self, input_features: list[str] = None):
        output_features = super().get_feature_names_out(input_features)
        return np.array([f.split('__')[-1] for f in output_features])

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
