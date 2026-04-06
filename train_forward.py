import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models import PINN, Sin
from losses import total_loss

# ========================= SETUP =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = np.load("data/params.npy", allow_pickle=True).item()
t_true = np.load("data/t_true.npy")
theta_true = np.load("data/theta_true.npy")
t_meas = np.load("data/t_meas.npy")
theta_meas_noisy = np.load("data/theta_meas_noisy.npy")

t_true_tensor = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32).to(device)
t_meas_tensor = torch.tensor(t_meas.reshape(-1, 1), dtype=torch.float32).to(device)
theta_meas_tensor = torch.tensor(theta_meas_noisy.reshape(-1, 1), dtype=torch.float32).to(device)

t0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
theta0 = torch.tensor([[data["theta0"]]], dtype=torch.float32).to(device)
omega0 = torch.tensor([[data["omega0"]]], dtype=torch.float32).to(device)

g = data["g"]
L = data["L"]
b_true = data["b_true"]

# More collocation points + random sampling (already good)
num_collocation = 15000
t_collocation = (torch.rand(num_collocation, 1) * 10.0).to(device)
t_collocation.requires_grad_(True)

# ========================= MODEL =========================
model = PINN(
    hidden_layers=4,
    neurons=64,
    activation=Sin,      # explicitly use the Sin you defined
    fix_b=True,
    b_true=b_true
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
print(f"Training on device: {device} | Fixed b = {model.b.item():.3f}")

# ========================= TRAINING =========================
num_epochs = 15000
os.makedirs("results/models", exist_ok=True)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    total_l, phys_l, data_l, ic_l = total_loss(
        model=model,
        t_collocation=t_collocation,
        t_meas=t_meas_tensor,
        theta_meas_noisy=theta_meas_tensor,
        t0=t0,
        theta0=theta0,
        omega0=omega0,
        g=g,
        L=L,
        lambda_physics=1.0,
        lambda_data=0.01,      # ← tiny but non-zero pressure
        lambda_ic=150.0        # ← this is the key fix
    )

    total_l.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:5d} | Total: {total_l.item():.2e} | Phys: {phys_l.item():.2e} | "
              f"Data: {data_l.item():.2e} | IC: {ic_l.item():.2e}")

print("Forward training finished!")

# Save
torch.save({'model_state_dict': model.state_dict()}, "results/models/forward_pinn_fixed.pth")

# ========================= PLOT =========================
model.eval()
with torch.no_grad():
    theta_pred = model(t_true_tensor).cpu().numpy().flatten()

plt.figure(figsize=(10, 6))
plt.plot(t_true, theta_true, 'b-', label='Ground Truth (SciPy)', linewidth=2)
plt.plot(t_true, theta_pred, 'r--', label='PINN Forward Prediction (fixed)', linewidth=2)
plt.scatter(t_meas, theta_meas_noisy, c='black', s=20, label='Noisy Measurements')
plt.xlabel('Time (s)')
plt.ylabel('Angle θ (radians)')
plt.title('Forward PINN – Corrected Version')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/plots/forward_pinn_result.png", dpi=200, bbox_inches='tight')
plt.show()

print("Plot saved as results/plots/forward_pinn_result.png")