# Physics-Informed Neural Network for Nonlinear Damped Pendulum
![Pendulum Animation](results/animation/pendulum_comparison.gif)

## Overview
This project implements a **Physics-Informed Neural Network (PINN)** that solves the nonlinear damped pendulum ODE in two complementary ways:

- **Forward problem**: Predicts the full motion θ(t) when all physical parameters are known.
- **Inverse problem**: Discovers the unknown damping coefficient \( b \) from only 40 noisy measurements while penalizing violations of the physical law through a soft residual loss.

The project demonstrates how AI can embed physical laws directly into the loss function and recover hidden parameters, a standard approach in scientific machine learning research.

## Physics Background
The system is governed by the nonlinear damped pendulum equation:

  $$\frac{d^2\theta}{dt^2} + b \frac{d\theta}{dt} + \frac{g}{L}\sin\theta = 0$$
  
Where:
- $\frac{d^2\theta}{dt^2}$: angular acceleration
- $b$: damping coefficient (friction/air resistance; the unknown parameter in the inverse problem)
- $\frac{d\theta}{dt}$: angular velocity
- $g = 9.81$ m/s²: gravitational acceleration 
- $L = 1$ m: pendulum length 

Initial conditions: $\theta(0) = 1.0$ rad (~57°), $\frac{d\theta}{dt}$(0) = 0.

## Methodology
1. **Data generation**: Synthetic ground-truth trajectory generated with SciPy’s `solve_ivp`. 40 noisy measurements (2 % Gaussian noise) simulate real lab data.
2. **Neural network**: MLP (the default implementation has five hidden linear layers) with sinusoidal (SIREN-style) activation and automatic differentiation to compute $\frac{d\theta}{dt}$ and $\frac{d^2\theta}{dt^2}$. ([models.py](models.py))
3. **Loss functions**: Physics residual + data fidelity + initial condition enforcement. ([losses.py](losses.py))
4. **Training**:
   - Forward: fixed $b = 0.25$, small data anchoring to avoid trivial solutions.
   - Inverse: $b$ is a learnable parameter; the network discovers its value from noisy data.

## Historical reported metrics (not independently reproduced)

The figures are committed, but trained checkpoints and raw training logs are not.
The values below are historical reports, not freshly verified measurements; the
current scripts alone cannot establish these exact values without retraining.
No superiority over a conventional ODE solver is claimed.

## Results - Forward Problem
Switching from Tanh to sinusoidal activation dramatically improved convergence and avoided the “trivial solution” problem in the reported run (collapse to equilibrium).
![Forward PINN (Tanh activation)](results/plots/forward_pinn_result_tanh.png)

**Final forward results** (after 15 000 epochs):
- Physics loss: ~1.5×10⁻⁴
- Excellent match to SciPy ground truth across the full 10 s interval.

![Forward PINN (Sin activation)](results/plots/forward_pinn_result.png)

A small data term (λ_data = 0.01) was used only to stabilize training.

## Results - Inverse Problem
The inverse PINN discovers the unknown damping coefficient while minimizing a soft ODE-residual penalty.

**Final inverse results** (after 20 000 epochs):
- Learned damping: \( b = 0.2527 \) (true value = 0.25)
- Relative error: **1.08 %**
- Motion prediction remains excellent despite noisy input data.

![Inverse PINN](results/plots/inverse_pinn_result.png)
![Inverse PINN b learning curve](results/plots/b_learning_curve.png)

The parameter $b$ converged rapidly in the first ~2 000 epochs and then stabilized, with minor oscillations due to the 2 % measurement noise.

## Quantitative Metrics
| Metric                  | Forward Problem | Inverse Problem |
|-------------------------|-----------------|-----------------|
| Physics loss (final)    | 1.5×10⁻⁴        | 8.45×10⁻⁵       |
| Mean Absolute Error (θ) | ~0.001 rad      | ~0.002 rad      |
| RMSE (θ)                | ~0.002 rad      | ~0.003 rad      |
| Learned b              | 0.25 (fixed)    | ~0.2527         |
| b relative error        | -               | ~1.08 %         |

## How to Run

Use Python 3.11+ from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data_generation.py
python train_forward.py
python train_inverse.py
```
## Future Work / Ablation Studies
* Test different network depths / widths
* Vary noise levels (1 % → 5 %)
* Compare Tanh vs. Sin activation quantitatively
* Add learning-rate scheduling and adaptive loss weighting
* Extend to double pendulum or chaotic regimes

## Acknowledgments / What I Learned
This project taught me how to embed physical laws into neural networks, 
handle inverse problems with noisy data, diagnose and fix common PINN failure modes 
(trivial solutions), and produce reproducible scientific ML research. 
It strengthened my skills in PyTorch, automatic differentiation, and computational physics.

## Demonstration without retraining

Open [the committed comparison animation](results/animation/pendulum_comparison.gif)
in a browser. This is a recorded demonstration of the reported forward prediction
against synthetic SciPy ground truth, not fresh model inference. The plotting script
`animate_pendulum.py` needs `results/models/forward_pinn_fixed.pth` and
`results/models/inverse_pinn.pth`; neither checkpoint is distributed in this repository.
Do not run that script expecting a fresh clone to contain trained models.
`data/` contains synthetic data only: 40 irregularly sampled observations, with Gaussian
noise scaled to 2% of each clean observation's absolute value.

## Contributions and assistance

Tommy originated this independent learning experiment before formal IOAI training
and substantially implemented it himself. AI was consulted on implementation details,
with very little AI-generated code. This is an introduction to scientific machine
learning, not a novel numerical method. PyTorch, SciPy, NumPy, Matplotlib and Pillow
provide the external software used here; see their imports and `requirements.txt`.
Codex assisted with the November 2026 documentation and portability corrections;
that assistance does not describe the authorship of the original experiment.
