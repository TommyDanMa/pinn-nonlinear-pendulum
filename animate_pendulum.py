import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import os
from models import PINN, Sin

# ========================= LOAD DATA & MODELS =========================
t_true = np.load("data/t_true.npy")
theta_true = np.load("data/theta_true.npy")

# Load forward model (best visual quality)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_forward = PINN(hidden_layers=4, neurons=64, activation=Sin, fix_b=True, b_true=0.25).to(device)
checkpoint = torch.load("results/models/forward_pinn_fixed.pth", map_location=device, weights_only=True)
model_forward.load_state_dict(checkpoint['model_state_dict'])
model_forward.eval()

# Load inverse model (for comparison)
model_inverse = PINN(hidden_layers=4, neurons=64, activation=Sin, fix_b=False).to(device)
checkpoint_inv = torch.load("results/models/inverse_pinn.pth", map_location=device, weights_only=True)
model_inverse.load_state_dict(checkpoint_inv['model_state_dict'])
model_inverse.eval()

t_tensor = torch.tensor(t_true.reshape(-1, 1), dtype=torch.float32).to(device)

with torch.no_grad():
    theta_pred_forward = model_forward(t_tensor).cpu().numpy().flatten()
    theta_pred_inverse = model_inverse(t_tensor).cpu().numpy().flatten()

# ========================= ANIMATION SETUP =========================
L = 1.0
frames = 400
fps = 30

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Nonlinear Damped Pendulum: Ground Truth vs PINN Predictions', fontsize=16)

# Setup both axes
for ax in (ax1, ax2):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

ax1.set_title('Ground Truth (SciPy)')
ax2.set_title('PINN Forward Prediction')

# Pendulum elements - Ground Truth
line_gt, = ax1.plot([], [], 'o-', lw=4, color='blue')
bob_gt = ax1.scatter([], [], s=300, color='red', zorder=5)
pivot = ax1.scatter([0], [0], s=80, color='black')

# Pendulum elements - PINN
line_pinn, = ax2.plot([], [], 'o-', lw=4, color='red')
bob_pinn = ax2.scatter([], [], s=300, color='red', zorder=5)
pivot2 = ax2.scatter([0], [0], s=80, color='black')

def init():
    line_gt.set_data([], [])
    bob_gt.set_offsets(np.empty((0, 2)))
    line_pinn.set_data([], [])
    bob_pinn.set_offsets(np.empty((0, 2)))
    return line_gt, bob_gt, line_pinn, bob_pinn

def animate(i):
    idx = int(i * len(t_true) / frames)
    theta_gt = theta_true[idx]
    theta_pinn = theta_pred_forward[idx]   # Change to theta_pred_inverse if you prefer

    # Ground Truth
    x_gt = L * np.sin(theta_gt)
    y_gt = -L * np.cos(theta_gt)
    line_gt.set_data([0, x_gt], [0, y_gt])
    bob_gt.set_offsets([[x_gt, y_gt]])

    # PINN Prediction
    x_pinn = L * np.sin(theta_pinn)
    y_pinn = -L * np.cos(theta_pinn)
    line_pinn.set_data([0, x_pinn], [0, y_pinn])
    bob_pinn.set_offsets([[x_pinn, y_pinn]])

    return line_gt, bob_gt, line_pinn, bob_pinn

ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=True, interval=30)

# Save as GIF
os.makedirs("results/animation", exist_ok=True)
writer = PillowWriter(fps=fps)
ani.save("results/animation/pendulum_comparison.gif", writer=writer, dpi=140)

print("Comparison animation saved as results/animation/pendulum_comparison.gif")
plt.close()