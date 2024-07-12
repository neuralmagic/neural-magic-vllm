from ._custom_classes import ScalarType

s4 = ScalarType.s(4, None)
u4 = ScalarType.u(4, None)
s8 = ScalarType.s(8, None)
u8 = ScalarType.u(8, None)
fE3M4 = ScalarType.f(4, 3)
fE4M3 = ScalarType.f(3, 4)
fE8M7 = ScalarType.f(7, 8)
fE5M10 = ScalarType.f(5, 11)

# colloquial names
bfloat16 = fE8M7
float16 = fE5M10
