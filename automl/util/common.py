# -*- encoding: utf-8 -*-

from typing import Optional

import numpy as np

__all__ = [
    'check_none',
    'check_for_bool',
    'check_false',
    'check_true',
    'resolve_factor',
    'HANDLES_MULTICLASS',
    'HANDLES_NUMERIC',
    'HANDLES_NOMINAL',
    'HANDLES_MISSING',
    'HANDLES_NOMINAL_CLASS'
]

HANDLES_MULTICLASS = 'handles_multiclass'
HANDLES_NUMERIC = 'handles_numeric'
HANDLES_NOMINAL = 'handles_nominal'
HANDLES_MISSING = 'handles_missing'
HANDLES_NOMINAL_CLASS = 'handles_nominal_class'


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
