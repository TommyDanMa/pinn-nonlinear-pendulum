import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import os

# ========================= PARAMETERS =========================
g = 9.81          # gravity (m/s²)
L = 1.0           # length of pendulum (m)
b_true = 0.25     # true damping coefficient (this is what inverse PINN will try to discover)
theta0 = 1.0      # initial angle (radians, ~57° — large enough for nonlinear behavior)
omega0 = 0.0      # initial angular velocity

t_start = 0.0
t_end = 10.0      # simulate for 10 seconds
num_points_true = 1000   # points for "ground truth" curve (for comparison & plotting)

# For inverse problem: number of "lab measurements"
num_measurements = 40
noise_level = 0.02    # 2% relative noise (realistic for simple sensor)

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========================= DATA GENERATION =========================
def pendulum_ode(t, y, b, g, L):
    """Right-hand side of the damped nonlinear pendulum ODE:
       d²θ/dt² + b dθ/dt + (g/L) sinθ = 0
       y = [dθ/dt, θ]"""
    theta = y[1]
    dtheta_dt = y[0]
    ddtheta_dt = -b * dtheta_dt - (g / L) * np.sin(theta)
    return [ddtheta_dt, dtheta_dt]

# Solve the ODE with SciPy to get "perfect" ground truth
sol = solve_ivp(
    fun=lambda t, y: pendulum_ode(t, y, b_true, g, L),
    t_span=(t_start, t_end),
    y0=[omega0, theta0],
    t_eval=np.linspace(t_start, t_end, num_points_true),
    rtol=1e-8,
    atol=1e-8
)

t_true = sol.t
theta_true = sol.y[1]       # angle θ(t)

print(f"Ground truth data generated: {len(t_true)} points")
print(f"True damping coefficient b = {b_true}")

# Create noisy "lab measurements" for the inverse problem
# Randomly sample some times (simulate taking measurements at irregular intervals)
measurement_indices = np.sort(np.random.choice(num_points_true, num_measurements, replace=False))
t_meas = t_true[measurement_indices]
theta_meas_clean = theta_true[measurement_indices]

# Add realistic Gaussian noise (relative to the signal amplitude)
noise = noise_level * np.abs(theta_meas_clean) * np.random.randn(num_measurements)
theta_meas_noisy = theta_meas_clean + noise

print(f"Noisy measurements created: {num_measurements} points with {noise_level*100:.0f}% noise")

# ========================= SAVE DATA =========================
os.makedirs("data", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

np.save("data/t_true.npy", t_true)
np.save("data/theta_true.npy", theta_true)
np.save("data/t_meas.npy", t_meas)
np.save("data/theta_meas_noisy.npy", theta_meas_noisy)

# Also save parameters for easy loading later
params = {
    "g": g,
    "L": L,
    "b_true": b_true,
    "theta0": theta0,
    "omega0": omega0,
    "t_start": t_start,
    "t_end": t_end,
    "noise_level": noise_level
}
np.save("data/params.npy", params)

print("All data saved in ./data/ folder")

# ========================= QUICK VISUALIZATION =========================
plt.figure(figsize=(10, 6))
plt.plot(t_true, theta_true, 'b-', label='Ground truth (SciPy)', linewidth=2)
plt.scatter(t_meas, theta_meas_noisy, c='red', s=30, label='Noisy lab measurements', zorder=5)
plt.xlabel('Time (s)')
plt.ylabel('Angle θ (radians)')
plt.title('Nonlinear Damped Pendulum — Ground Truth vs Noisy Measurements')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/plots/data_generation_preview.png", dpi=200, bbox_inches='tight')
plt.show()

print("Preview plot saved as results/plots/data_generation_preview.png")