import torch

print("Pytorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(1000, 1000, device=device)
y = torch.matmul(x, x)
print(f"Matrix computation successful on {device}!")