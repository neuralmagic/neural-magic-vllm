from typing import Union


class VLLMType:

    def __init__(self, mantissa: int, exponent: int, signed: bool) -> None:
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
