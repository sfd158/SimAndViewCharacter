"""
conclusion: I can use torch.as_tensor(, device=) directly.
"""
import numpy as np
import torch
import time

v = np.random.rand(8192 * 81 * 13 * 2).astype(np.float32)
w = np.random.rand(4096 * 27 * 13 * 2).astype(np.float32)
cuda_device = torch.device("cuda")
print(torch.cuda.is_available())

"""
start = time.time()
for i in range(2333):
    w = torch.from_numpy(v)
    w.to(cuda_device)
print(time.time() - start)

start = time.time()
for i in range(2333):
    w = torch.as_tensor(v, device=cuda_device)  # as tensor is faster than to for a single vector
print(time.time() - start)
"""

start = time.time()
for i in range(2333):
    tmp0 = torch.from_numpy(v)
    tmp0.to(cuda_device, non_blocking=True)
    tmp1 = torch.from_numpy(w)
    tmp1.to(cuda_device, non_blocking=True)

print(time.time() - start)

start = time.time()
for i in range(2333):
    tmp0 = torch.as_tensor(v, device=cuda_device)  # as tensor is faster than to for a single vector
    tmp1 = torch.as_tensor(w, device=cuda_device)
print(time.time() - start)