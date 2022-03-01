from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING, Optional, Any

from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from dswizard.components import util
from dswizard.components.base import EstimatorComponent, HasChildComponents
from dswizard.components.util import prefixed_name, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, \
    HANDLES_MISSING, HANDLES_NOMINAL_CLASS

if TYPE_CHECKING:
    from dswizard.components.meta_features import MetaFeaturesDict


class ConfigurablePipeline(Pipeline, EstimatorComponent, HasChildComponents):

    def __init__(self,
                 steps: List[Tuple[str, EstimatorComponent]],
                 configuration: Optional[Dict] = None):
        self.args = {'steps': [(label, util.serialize(comp)) for label, comp in steps], 'configuration': configuration}

        self.configuration = configuration

        # super.__init__ has to be called after initializing all properties provided in constructor
        super().__init__(steps, verbose=False)
        self.steps: List[Tuple[str, EstimatorComponent]] = self.steps  # only for type hinting
        self.steps_ = dict(steps)
        self.configuration_space: ConfigurationSpace = self.get_hyperparameter_search_space()

        self.fit_time = 0
        self.config_time = 0

        if configuration is not None:
            self.set_hyperparameters(configuration)

    @staticmethod
    def get_properties() -> Dict:
        return {'shortname': 'pipeline',
                'name': 'Configurable Pipeline',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True
                }

    def to_networkx(self, prefix: str = None):
        import networkx as nx
        g = nx.DiGraph()
        predecessor = None
        for name, estimator in self.steps_.items():
            name = prefixed_name(prefix, name)
            g.add_node(name, label=estimator.name().split('.')[-1], name=name)

            if predecessor is not None:
                g.add_edge(predecessor, name)
            predecessor = name
        return g

    def set_hyperparameters(self, configuration: Dict = None, init_params=None):
        self.configuration = configuration
        self.set_child_hyperparameters(self.steps, configuration, init_params)
        return self

    def get_hyperparameter_search_space(self, mf: Optional[MetaFeaturesDict] = None) -> ConfigurationSpace:
        return self.get_child_hyperparameter_search_space(self.steps, mf)

    def items(self):
        return self.steps_.items()

    def get_feature_names_out(self, input_features=None):
        feature_names_out = input_features
        for _, name, transform in self._iter():
            if not hasattr(transform, "get_feature_names_out"):
                raise AttributeError(
                    "Estimator {} does not provide get_feature_names_out. "
                    "Did you mean to call pipeline[:-1].get_feature_names_out"
                    "()?".format(name)
                )
            feature_names_out = transform.get_feature_names_out(feature_names_out)
            if hasattr(transform, "predict"):
                feature_names_out = [f"{name}__{f}" for f in feature_names_out]
        return feature_names_out

    @staticmethod
    def deserialize(steps: List[str, Dict[str, Any]], **kwargs) -> 'ConfigurablePipeline':
        steps_ = []
        for name, value in steps:
            steps_.append((name, util.deserialize(**value)))
        return ConfigurablePipeline(steps_, **kwargs)

    def __lt__(self, other: 'ConfigurablePipeline'):
        s1 = tuple(e.name() for e in self.steps_.values())
        s2 = tuple(e.name() for e in other.steps_.values())
        return s1 < s2

    def __copy__(self):
        return ConfigurablePipeline(clone(self.steps, safe=False), self.configuration)
