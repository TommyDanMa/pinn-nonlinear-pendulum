# Physics-Informed Neural Network for Nonlinear Damped Pendulum

## Overview
Short 2-3 sentence summary: what the project does and why it's interesting (forward + inverse problem, AI discovering physics parameters).

## Physics Background
- Write the ODE with LaTeX:  
  $$\frac{d^2\theta}{dt^2} + b \frac{d\theta}{dt} + \frac{g}{L}\sin\theta = 0$$
- Brief explanation of each term (you already understand this well).

## Methodology
- PINN architecture (reference models.py)
- Loss functions (physics, data, initial conditions)
- Training procedure

## Results - Forward Problem
- Plot of PINN vs SciPy
- Training loss curve
- Accuracy metrics (e.g. mean absolute error)

## Results - Inverse Problem (to be added later)
- Discovered value of b vs true b=0.25
- Tables and plots

## How to Run
```bash
python data_generation.py
python train_forward.py
python train_inverse.py
```
## Future Work / Ablation Studies
- Different network sizes
- Noise levels
- Different activations

## Acknowledgments / What I Learned
