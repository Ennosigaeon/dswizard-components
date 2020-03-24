# -*- encoding: utf-8 -*-

from typing import Optional

import numpy as np

__all__ = [
    'check_none',
    'check_for_bool',
    'check_false',
    'check_true',
    'resolve_factor'
]


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


def resolve_factor(value: Optional[float], n: float) -> Optional[int]:
    if check_none(value):
        return None
    else:
        return max(1, int(np.round(value * n, 0)))
