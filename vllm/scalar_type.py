from enum import Enum

from ._core_ext import ScalarType


# Mirrors enum in `core/scalar_type.hpp`
class NanRepr(Enum):
    NONE = 0  # nans are not supported
    IEEE_754 = 1  # nans are: Exp all 1s, mantissa not all 0s
    EXTD_RANGE_MAX_MIN = 2  # nans are: Exp all 1s, mantissa all 1s


# naming generally follows: https://github.com/jax-ml/ml_dtypes
# for floating point types (leading f):
#  - trailing f: means finite values only (no infinities)
#  - trailing n: means nans are supported
#  - no-trailing letters: means it follows IEEE 754 conventions
#  - E_: exponent size
#  - M_: mantissa size
# for integer types:
#  - leading s: means signed
#  - leading u: means unsigned
#  - number following s/u: number of bits
#  - bX: indicates a non-zero bias of X


class scalar_types:
    s4 = ScalarType.s(4, None)
    u4 = ScalarType.u(4, None)
    s8 = ScalarType.s(8, None)
    u8 = ScalarType.u(8, None)
    fE4M3fn = ScalarType.fn(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN.value)
    fE5M2 = ScalarType.f(5, 2)
    fE8M7 = ScalarType.f(8, 7)
    fE5M10 = ScalarType.f(5, 10)

    # fp6, https://github.com/usyd-fsalab/fp6_llm/tree/main
    fE3M2f = ScalarType.fn(3, 2, True, NanRepr.NONE.value)

    # "gptq" types
    u4b8 = ScalarType.u(4, 8)
    u8b128 = ScalarType.u(8, 128)

    # colloquial names
    bfloat16 = fE8M7
    float16 = fE5M10
