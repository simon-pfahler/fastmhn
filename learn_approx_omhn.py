import numpy as np
import torch

torch.set_grad_enabled(False)

import fastmhn

results_filename = "theta.dat"

d = 3
N = 100
data = np.random.randint(2, size=(N, d), dtype=np.int32)

# >>> optimization parameters
nr_iterations = 200
alpha = 0.1
beta1 = 0.7
beta2 = 0.9
eps = 1e-8
reg = 1e-2
# <<< optimization parameters

theta_np = np.zeros((d + 1, d))
theta_np[:d] = fastmhn.utility.create_indep_model(data)

theta = torch.tensor(theta_np, requires_grad=True)

optimizer = torch.optim.Adam([theta], lr=alpha, betas=(beta1, beta2), eps=eps)

for t in range(nr_iterations):
    optimizer.zero_grad()

    # create MHN theta matrix equivalent to current oMHN
    ctheta = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                ctheta[i, j] = theta[i, j]
            else:
                ctheta[i, j] = theta[i, j] - theta[d, j]

    g = torch.zeros(theta.shape, dtype=torch.double)
    g[:d] = -torch.from_numpy(
        fastmhn.approx.approx_gradient(
            ctheta, data, max_cluster_size=mcs, verbose=True
        )
    )

    # observation rate gradients
    g[d, :] = -torch.einsum("ij->j", g[:d] * (1 - torch.eye(g.shape[1])))

    # regularization
    g[:d] += reg * torch.sign(theta[:d] * (1 - torch.eye(theta.shape[1])))
    g[d, :] += reg * torch.sign(theta[d, :])

    theta.grad = g

    print(f"{t} - {torch.linalg.norm(theta.grad)}")

    optimizer.step()

with open(results_filename, "w") as f:
    for i in range(theta.shape[0]):
        f.write(" ".join(map(str, theta.numpy()[i])) + "\n")
