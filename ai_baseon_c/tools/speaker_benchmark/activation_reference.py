"""Independent NumPy binary32 reference for the portable activation contract.

This module imports no C runtime/header and calls no exp/tanh implementation.
Each arithmetic ufunc operates on float32 arrays; powers of two are constructed
from their IEEE bits. The mathematical recipe is split-ln2 reduction with a
degree-7 exponential polynomial, degree-13 odd tanh below 0.5, and direct
1/(1+exp(-x)) sigmoid. Correct rounding to libm is not claimed.
"""

import numpy as np

F32 = np.float32
CONTRACT = "deterministic_float32_v1"


def _values(values):
    return np.asarray(values, dtype=np.float32)


def exponential(values):
    values = _values(values)
    flat = values.reshape(-1)
    output = np.empty_like(flat)
    nan = np.isnan(flat)
    high, low = flat >= F32(88.72283935546875), flat <= F32(-104)
    output[nan] = np.asarray([0x7fc00000], dtype=np.uint32).view(np.float32)[0]
    output[high], output[low] = F32(np.inf), F32(0)
    active = ~(nan | high | low)
    x = flat[active]
    scaled = x * F32(1.4426950408889634)
    integer = np.trunc(scaled + np.where(scaled < F32(0), F32(-.5), F32(.5))).astype(np.int32)
    n = integer.astype(np.float32)
    reduced_hi = x - n * F32(.693145751953125)
    reduced = reduced_hi - n * F32(.000001428606765330187)
    p = np.full_like(reduced, F32(.0001984126984126984))
    for coefficient in (.001388888888888889, .008333333333333333,
                        .041666666666666664, .16666666666666666, .5, 1., 1.):
        p = F32(coefficient) + reduced * p
    y = np.empty_like(p)
    normal = (integer >= -126) & (integer <= 127)
    powers = ((integer[normal] + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)
    y[normal] = p[normal] * powers
    upper = integer > 127
    y[upper] = (p[upper] * F32(2.0 ** 127)) * F32(2)
    lower = integer < -126
    powers = ((integer[lower] + 24 + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)
    y[lower] = (p[lower] * powers) * F32(2.0 ** -24)
    output[active] = y
    return output.reshape(values.shape)


def hyperbolic_tangent(values):
    values = _values(values)
    absolute = np.abs(values)
    output = np.empty_like(values)
    small = absolute <= F32(.5)
    x = absolute[small]
    square = x * x
    p = np.full_like(x, F32(.003592128036572481))
    for coefficient in (-.008863235529902197, .021869488536155203, -.05396825396825397,
                        .13333333333333333, -.33333333333333333):
        p = F32(coefficient) + square * p
    output[small] = x + (x * square) * p
    large = absolute >= F32(10)
    output[large] = F32(1)
    active = ~(small | large | np.isnan(values))
    e = exponential(F32(-2) * absolute[active])
    output[active] = (F32(1) - e) / (F32(1) + e)
    output = np.asarray(np.copysign(output, values), dtype=np.float32)
    output[np.isnan(values)] = np.asarray([0x7fc00000], dtype=np.uint32).view(np.float32)[0]
    return output


def logistic(values):
    values = _values(values)
    e = exponential(-values)
    denominator = F32(1) + e
    output = np.asarray(F32(1) / denominator, dtype=np.float32)
    output[np.isnan(values)] = np.asarray([0x7fc00000], dtype=np.uint32).view(np.float32)[0]
    return output.astype(np.float32, copy=False)


class PortableMath:
    """Scalar-math interface used by the independent recurrent integer oracle."""

    contract = CONTRACT
    command = None

    def apply(self, name, values):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return {"exp": exponential, "tanh": hyperbolic_tangent, "sigmoid": logistic}[name](values)

    def close(self):
        pass
