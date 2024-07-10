from ._custom_classes import VLLMType

int4 = VLLMType(3, 0, True)
uint4 = VLLMType(4, 0, False)
int8 = VLLMType(7, 0, True)
uint8 = VLLMType(8, 0, False)
fE3M4 = VLLMType(4, 3, True)
fE4M3 = VLLMType(3, 4, True)
fE8M7 = VLLMType(7, 8, True)
fE5M10 = VLLMType(5, 11, True)

# colloquial names
bfloat16 = fE8M7
float16 = fE5M10
