# -*- encoding: utf-8 -*-
import importlib
import inspect
from typing import Optional, Dict

import numpy as np

HANDLES_MULTICLASS = 'handles_multiclass'
HANDLES_NUMERIC = 'handles_numeric'
HANDLES_NOMINAL = 'handles_nominal'
HANDLES_MISSING = 'handles_missing'
HANDLES_NOMINAL_CLASS = 'handles_nominal_class'


def get_type(clazz: str) -> type:
    module_name = clazz.rpartition(".")[0]
    class_name = clazz.split(".")[-1]

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def serialize(object_) -> Dict:
    try:
        return object_.serialize()
    except AttributeError:
        cls = object_.__class__
        return {'clazz': '.'.join([cls.__module__, cls.__qualname__]), 'args': {}}


def deserialize(clazz: str, args=None):
    if args is None:
        args = {}

    type_ = get_type(clazz)
    try:
        inspect.getattr_static(type_, "deserialize")
        # noinspection PyUnresolvedReferences
        return type_.deserialize(**args)
    except AttributeError:
        return type_(**args)


def check_true(p):
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p):
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p):
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p):
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError("%s is not a bool" % str(p))


def resolve_factor(value: Optional[float], n: float, default=None, cs_default=None) -> Optional[int]:
    if check_none(value) or value == cs_default:
        return default
    else:
        return max(1, int(np.round(value * n, 0)))


def softmax(df):
    if len(df.shape) == 1:
        df[df > 20] = 20
        df[df < -20] = -20
        ppositive = 1 / (1 + np.exp(-df))
        ppositive[ppositive > 0.999999] = 1
        ppositive[ppositive < 0.0000001] = 0
        return np.transpose(np.array((1 - ppositive, ppositive)))
    else:
        # Compute the Softmax like it is described here:
        # http://www.iro.umontreal.ca/~bengioy/dlbook/numerical.html
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


def convert_multioutput_multiclass_to_multilabel(probas):
    if isinstance(probas, np.ndarray) and len(probas.shape) > 2:
        raise ValueError('New unsupported sklearn output!')
    if isinstance(probas, list):
        multioutput_probas = np.ndarray((probas[0].shape[0], len(probas)))
        for i, output in enumerate(probas):
            if output.shape[1] > 2:
                raise ValueError('Multioutput-Multiclass supported by '
                                 'scikit-learn, but not by auto-sklearn!')
            # Only copy the probability of something having class 1
            elif output.shape[1] == 2:
                multioutput_probas[:, i] = output[:, 1]
            # This label was never observed positive in the training data,
            # therefore it is only the probability for the label being False
            else:
                multioutput_probas[:, i] = 0
        probas = multioutput_probas
    return probas


def prefixed_name(prefix: Optional[str], name: str) -> str:
    """
    Returns the potentially prefixed name name.
    """
    return name if prefix is None else f'{prefix}:{name}'


def object_log(X: np.ndarray):
    return np.log(X.astype(float))
