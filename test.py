import torch
import time
import vllm
from vllm._custom_ops import cutlass_scaled_mm
m = 4
n = 4096
k = 6144
dtype = torch.float8_e4m3fn
out_dtype = torch.float16

a = torch.empty(m, k).normal_(mean=0.0, std=0.5).to(dtype=dtype, device='cuda')
bt = torch.empty(n, k).normal_(mean=0.0, std=0.5).to(dtype=dtype, device='cuda').t()
scale_a = torch.ones((m,1)).to(dtype=torch.float32, device='cuda')
scale_b = torch.ones((1,n)).to(dtype=torch.float32, device='cuda')
bias = torch.empty((2,)).normal_(mean=0.0, std=0.5).to(dtype=torch.float16, device='cuda')

y = cutlass_scaled_mm(a, bt, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b)
torch.cuda.synchronize()
print("ok bias0")
y = cutlass_scaled_mm(a, bt, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias)
torch.cuda.synchronize()
print("ok bias1")