import numpy as np
import torch

import fastmhn

torch.set_grad_enabled(False)

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

# >>> Print information about dataset
avg_MB = np.mean(np.sum(data, axis=1))
max_MB = np.max(np.sum(data, axis=1))
nr_samples_approx = np.sum(np.sum(data, axis=1) > mcs)
print(
    f"Dataset information:\n"
    f"\t{data.shape[0]} Patients\n"
    f"\tAverage mutational burden: {avg_MB}\n"
    f"\tMaximum mutational burden: {max_MB}\n"
    f"\tNumber of samples with MB > {mcs}: {nr_samples_approx}"
)
# <<< Print information about dataset

theta = torch.tensor(
    fastmhn.utility.create_indep_model(data), requires_grad=True
)

optimizer = torch.optim.Adam([theta], lr=alpha, betas=(beta1, beta2), eps=eps)

# training loop
for t in range(nr_iterations):
    optimizer.zero_grad()

    g = -torch.from_numpy(
        fastmhn.approx.approx_gradient_and_score(theta.numpy(), data)[0]
    )
    g += reg * torch.sign(theta * (1 - torch.eye(theta.shape[0])))

    theta.grad = g

    print(f"{t} - {torch.linalg.norm(theta.grad)}")

    optimizer.step()

np.savetxt(results_filename, theta.numpy())
