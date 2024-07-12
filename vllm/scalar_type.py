from ._custom_classes import ScalarType

s4 = ScalarType(3, 0, 0, True)
u4 = ScalarType(4, 0, 0, False)
i8 = ScalarType(7, 0, 0, True)
u8 = ScalarType(8, 0, 0, False)
fE3M4 = ScalarType(4, 3, 0, True)
fE4M3 = ScalarType(3, 4, 0, True)
fE8M7 = ScalarType(7, 8, 0, True)
fE5M10 = ScalarType(5, 11, 0, True)

# colloquial names
bfloat16 = fE8M7
float16 = fE5M10
