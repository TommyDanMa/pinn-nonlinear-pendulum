import torch
import torch.nn as nn
import numpy as np


def physics_loss(model, t_collocation, g, L):
    """
    Computes how much the predicted motion violates the pendulum ODE.
    This is the core of the PINN.
    """
    theta, dtheta_dt, ddtheta_ddt = model.get_derivatives(t_collocation)

    # Residual of the ODE: ddtheta + b * dtheta + (g/L) * sin(theta) should be ~0
    residual = ddtheta_ddt + model.b * dtheta_dt + (g / L) * torch.sin(theta)

    # Mean squared residual (we want this to go to zero)
    loss_physics = torch.mean(residual ** 2)
    return loss_physics


def data_loss(model, t_meas, theta_meas_noisy):
    """
    How well the network fits the noisy "lab" measurements (only used in inverse problem)
    """
    theta_pred = model.forward(t_meas)  # shape [N, 1]
    loss_data = torch.mean((theta_pred - theta_meas_noisy) ** 2)
    return loss_data


def initial_condition_loss(model, t0, theta0, omega0):
    """
    Enforce the known initial conditions at t=0
    """
    theta, dtheta_dt, _ = model.get_derivatives(t0)

    loss_ic_theta = torch.mean((theta - theta0) ** 2)
    loss_ic_omega = torch.mean((dtheta_dt - omega0) ** 2)

    return loss_ic_theta + loss_ic_omega


def total_loss(model, t_collocation, t_meas, theta_meas_noisy,
               t0, theta0, omega0, g, L,
               lambda_physics=1.0, lambda_data=1.0, lambda_ic=1.0):
    """
    Combines all three losses.
    lambda_* are weights that control the importance of each term.
    """
    loss_phys = physics_loss(model, t_collocation, g, L)
    loss_data = data_loss(model, t_meas, theta_meas_noisy)
    loss_ic = initial_condition_loss(model, t0, theta0, omega0)

    total = lambda_physics * loss_phys + lambda_data * loss_data + lambda_ic * loss_ic
    return total, loss_phys, loss_data, loss_ic