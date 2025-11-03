# -------------------------
# Import
# -------------------------
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import time

# -------------------------
# Triton Softmax Kernel
# -------------------------
@triton.jit
def softmax_kernel(X_ptr, Y_ptr, B, D, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < D

    x_ptr = X_ptr + row_idx * D + col_idx
    y_ptr = Y_ptr + row_idx * D + col_idx

    # Load input
    x_vals = tl.load(x_ptr, mask=mask)

    # 1️⃣ Compute row max
    row_max = tl.max(x_vals, axis=0)

    # 2️⃣ Compute sum(exp(x - max))
    x_exp = tl.exp(x_vals - row_max)
    row_sum = tl.sum(x_exp, axis=0)

    # 3️⃣ Write softmax output
    tl.store(y_ptr, x_exp / row_sum, mask=mask)


# -------------------------
# Python wrapper
# -------------------------
def softmax_triton(x: torch.Tensor):
    B, D = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024  # 每 block 处理的元素数
    grid = (B,)
    softmax_kernel[grid](x, y, B, D, BLOCK_SIZE=BLOCK_SIZE)
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
y_gpu = softmax_triton(x)
torch.cuda.synchronize()
end = time.time()
print("GPU Triton softmax time: {:.6f} s".format(end - start))

# Max error
max_err = (y_cpu.to('cuda') - y_gpu).abs().max()
print("Max error:", max_err.item())

# -------------------------
# Optional: print some outputs
# -------------------------
print("First row CPU:", y_cpu[0][:10].numpy())
print("First row GPU:", y_gpu[0][:10].cpu().numpy())
