# -------------------------
# Imports
# -------------------------
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import time

# -------------------------
# Triton Softmax Kernel (分块大 hidden_dim)
# -------------------------
@triton.jit
def softmax_kernel(X_ptr, Y_ptr, B, D, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < D

    # 指针
    x_ptr = X_ptr + row_idx * D + offset
    y_ptr = Y_ptr + row_idx * D + offset

    # 1️⃣ load input
    x_vals = tl.load(x_ptr, mask=mask)

    # 2️⃣ compute max
    max_val = tl.max(x_vals, axis=0)

    # 3️⃣ compute exp(x - max)
    x_exp = tl.exp(x_vals - max_val)

    # 4️⃣ sum
    sum_exp = tl.sum(x_exp, axis=0)

    # 5️⃣ softmax
    tl.store(y_ptr, x_exp / sum_exp, mask=mask)


# -------------------------
# Python wrapper
# -------------------------
def softmax_triton(x: torch.Tensor, block_size=1024):
    B, D = x.shape
    y = torch.empty_like(x)
    grid = (B,)
    softmax_kernel[grid](x, y, B, D, BLOCK_SIZE=block_size)
    return y

# -------------------------
# Test
# -------------------------
B, D = 512, 2048
x = torch.randn(B, D, device='cuda')

# CPU Softmax
x_cpu = x.cpu()
start = time.time()
y_cpu = F.softmax(x_cpu, dim=1)
end = time.time()
print("CPU softmax time: {:.6f} s".format(end - start))

# GPU Triton Softmax
torch.cuda.synchronize()
start = time.time()
y_gpu = softmax_triton(x, block_size=1024)
torch.cuda.synchronize()
end = time.time()
print("GPU Triton softmax time: {:.6f} s".format(end - start))

# Max error
max_err = (y_cpu.to('cuda') - y_gpu).abs().max()
print("Max error:", max_err.item())

# 输出前 10 个元素做检查
print("First row CPU:", y_cpu[0][:10].numpy())
print("First row GPU:", y_gpu[0][:10].cpu().numpy())
