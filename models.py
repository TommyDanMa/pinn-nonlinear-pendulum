import torch
import torch.nn as nn

class Sin(nn.Module):
    """Custom sine activation (required because nn.Sin does not exist)"""
    def forward(self, x):
        return torch.sin(x)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for the nonlinear damped pendulum.
    Input: time t (scalar or tensor of shape [N, 1])
    Output: theta(t)  (angle at that time)
    """

    def __init__(self, hidden_layers=4, neurons=64, activation=Sin, fix_b=False, b_true=0.25):
        super().__init__()

        self.activation = activation

        # ====================== NETWORK ======================

        # Input layer: t → hidden
        layers = [nn.Linear(1, neurons)]

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(activation())
            layers.append(nn.Linear(neurons, neurons))

        # Output layer: hidden → theta (1 value)
        layers.append(activation())
        layers.append(nn.Linear(neurons, 1))

        self.net = nn.Sequential(*layers)

        # ====================== SIREN INITIALIZATION ======================
        with torch.no_grad():
            # First layer: scale by 1 (input is just t)
            nn.init.uniform_(self.net[0].weight, -1.0, 1.0)
            self.net[0].bias.data.uniform_(-1.0 / 1.0, 1.0 / 1.0)  # omega_0 = 1 for first layer

            # Hidden layers: scale by sqrt(6 / fan_in) * omega_0
            for i in range(1, len(self.net), 2):  # every Linear layer
                if isinstance(self.net[i], nn.Linear):
                    fan_in = self.net[i].weight.shape[1]
                    nn.init.uniform_(self.net[i].weight,
                                     -torch.sqrt(torch.tensor(6.0 / fan_in)),
                                     torch.sqrt(torch.tensor(6.0 / fan_in)))
                    self.net[i].bias.data.uniform_(-1.0, 1.0)

        # ====================== DAMPING PARAMETER ======================
        if fix_b:
            self.register_buffer('b', torch.tensor([b_true], dtype=torch.float32))
        else:
            self.b = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, t):
        """
        Forward pass: predict theta(t)
        t should be a tensor of shape [N, 1]
        """
        return self.net(t)

    def get_derivatives(self, t):
        """
        Compute first and second derivatives using automatic differentiation.
        This is the key feature that allows the network to 'know' physics.
        Returns: theta, dtheta_dt, ddtheta_ddt
        """
        t.requires_grad_(True)  # Enable gradient tracking for t

        theta = self.forward(t)  # Predict theta(t)

        # First derivative: dtheta/dt
        dtheta_dt = torch.autograd.grad(
            outputs=theta,
            inputs=t,
            grad_outputs=torch.ones_like(theta),
            create_graph=True,  # Need this for second derivative
            retain_graph=True
        )[0]

        # Second derivative: d²theta/dt²
        ddtheta_ddt = torch.autograd.grad(
            outputs=dtheta_dt,
            inputs=t,
            grad_outputs=torch.ones_like(dtheta_dt),
            create_graph=True
        )[0]

        return theta, dtheta_dt, ddtheta_ddt


# Simple test to make sure the model loads without errors
if __name__ == "__main__":
    model = PINN(hidden_layers=3, neurons=50)
    print("Model created successfully!")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Learnable damping b initialized to: {model.b.item():.4f}")

    # Quick test forward pass
    test_t = torch.tensor([[0.0], [1.0], [5.0]], dtype=torch.float32)
    theta_pred = model(test_t)
    print(f"Test prediction shape: {theta_pred.shape}")
    print("Everything looks good!")