import numpy as np
import torch
from fastmhn.utility import create_indep_model
from fastmhn.approx import approx_gradient
from fastmhn.explicit import create_full_Q, calculate_pTheta

def optimize_theta(d=8, N=300, nr_iterations=200, reg=1e-2):
    data = np.random.randint(2, size=(N, d), dtype=np.int32)
    theta = torch.tensor(create_indep_model(data), requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.1, betas=(0.7, 0.9), eps=1e-8)

    for _ in range(nr_iterations):
        optimizer.zero_grad()
        g = -torch.from_numpy(approx_gradient(theta.detach().numpy(), data))
        g += reg * torch.sign(theta * (1 - torch.eye(theta.shape[0])))
        theta.grad = g
        optimizer.step()

    theta_np = theta.detach().numpy()
    Q_full = create_full_Q(theta_np)
    p_full = calculate_pTheta(theta_np)

    return theta_np, Q_full, p_full