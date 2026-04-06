import torch

print("Pytorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

x = torch.rand(1000, 1000, device="cuda")
y = torch.matmul(x, x)
print("GPU computation succesful!")