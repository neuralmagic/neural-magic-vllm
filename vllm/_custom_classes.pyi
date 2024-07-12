from typing import Union, Optional


class ScalarType:

    def __init__(self, mantissa: int, exponent: int, bias: int, signed: bool) -> None:
        ...
        
    @classmethod
    def s(cls, size_bits: int, bias: Optional[int]) -> ScalarType:
        ...

    @classmethod
    def u(cls, size_bits: int, bias: Optional[int]) -> ScalarType:
        ...
        
    @classmethod
    def f(cls, mantissa: int, exponent: int) -> ScalarType:
        ...

    @property
    def mantissa(self) -> int:
        ...

    @property
    def exponent(self) -> int:
        ...

    @property
    def size_bits(self) -> int:
        ...

    def max(self) -> Union[int, float]:
        ...

    def min(self) -> Union[int, float]:
        ...

    def is_signed(self) -> bool:
        ...

    def is_integer(self) -> bool:
        ...

    def is_floating_point(self) -> bool:
        ...

    def __eq__(self, value: object) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...
