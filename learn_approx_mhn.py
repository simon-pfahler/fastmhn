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

theta = torch.tensor(
    fastmhn.utility.create_indep_model(data), requires_grad=True
)

optimizer = torch.optim.Adam([theta], lr=alpha, betas=(beta1, beta2), eps=eps)

for t in range(nr_iterations):
    optimizer.zero_grad()

    g = -torch.from_numpy(fastmhn.approx.approx_gradient(theta.numpy(), data))
    g += reg * torch.sign(theta * (1 - torch.eye(theta.shape[0])))

    theta.grad = g

    print(f"{t} - {torch.linalg.norm(theta.grad)}")

    optimizer.step()

with open(results_filename, "w") as f:
    for i in range(theta.shape[0]):
        f.write(" ".join(map(str, theta.numpy()[i])) + "\n")
