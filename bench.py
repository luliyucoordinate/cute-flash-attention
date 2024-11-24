import os 
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from flashinfer import single_prefill_with_kv_cache
from flash_attn import flash_attn_func

# Add a new environment variable  
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

REMOVE_NVCC_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]
for flag in REMOVE_NVCC_FLAGS:
    try:
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass


torch.manual_seed(0)
# Load the CUDA kernel as a python module
myflash = load(name='myflash', 
                    sources=[
                        'main.cpp', 
                        'flash.cu', 
                    ], 
                    extra_cuda_cflags=[
                        '-O2', 
                        '-lcublas', 
                        '-lcublasLt', 
                        '-std=c++17', 
                        '-I/mnt/d/cuda/cutlass/include', 
                        '-I/mnt/d/cuda/cutlass/tools/util/include', 
                    ], 
                )

def manual_attn(q, k, v, attn_mask=None):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if attn_mask != None:
        att.masked_fill_(attn_mask, float('-inf'))  # Apply mask
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

batch_size = 1
n_head = 32
q_len = 1024 
kv_len = q_len
head_embd = 64 

q = torch.randn(batch_size, q_len, n_head, head_embd).cuda().half()
k = torch.randn(batch_size, kv_len, n_head, head_embd).cuda().half()
v = torch.randn(batch_size, kv_len, n_head, head_embd).cuda().half()
q1 = q.transpose(1, 2).contiguous()
k1 = k.transpose(1, 2).contiguous()
v1 = v.transpose(1, 2).contiguous()
q2 = q.reshape(batch_size * q_len, n_head, head_embd)
k2 = k.reshape(batch_size * kv_len, n_head, head_embd)
v2 = v.reshape(batch_size * kv_len, n_head, head_embd)

a = manual_attn(q1, k1, v1)
b = myflash.forward(q1, k1, v1)
c = single_prefill_with_kv_cache(q2, k2, v2)
d = flash_attn_func(q, k, v)
print('attn values sanity check:', torch.allclose(a, b, rtol=1e-01, atol=1e-02))