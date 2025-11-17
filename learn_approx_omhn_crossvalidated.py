import numpy as np
import torch

import fastmhn

torch.set_grad_enabled(False)

results_filename = "theta.dat"

d = 3
N = 100
data = np.random.randint(2, size=(N, d), dtype=np.int32)

# >>> optimization parameters
nr_iterations = 50
alpha = 0.1
beta1 = 0.7
beta2 = 0.9
eps = 1e-8
reg = 1e-2
mcs = 25
k = 5  # level of cross-validation
# <<< optimization parameters

# >>> shuffle data
rng = np.random.default_rng(42)
rng.shuffle(data, axis=0)
# <<< shuffle data

# >>> get fold sizes for cross-validation
fold_sizes = (N // k) * np.ones(k, dtype=int)
fold_sizes[: N % k] += 1
# <<< get fold sizes for cross-validation

average_validation_score = 0

# loop over all subsamples
for k_index in range(k):
    # >>> get validation and training datasets
    data_val = data[
        np.sum(fold_sizes[:k_index]) : np.sum(fold_sizes[: k_index + 1])
    ]
    data_train = np.concatenate(
        (
            data[: np.sum(fold_sizes[:k_index])],
            data[np.sum(fold_sizes[: k_index + 1]) :],
        )
    )
    # <<< get validation and training datasets

    theta_np = np.zeros((d + 1, d))
    theta_np[:d] = fastmhn.utility.create_indep_model(data_train)

    theta = torch.tensor(theta_np, requires_grad=True)

    optimizer = torch.optim.Adam(
        [theta], lr=alpha, betas=(beta1, beta2), eps=eps
    )

    for t in range(nr_iterations):
        optimizer.zero_grad()

        # create MHN theta matrix equivalent to current oMHN
        ctheta = fastmhn.utility.cmhn_from_omhn(theta)

        g = torch.zeros(theta.shape, dtype=torch.double)
        g[:d] = -torch.from_numpy(
            fastmhn.approx.approx_gradient_and_score(
                ctheta, data_train, max_cluster_size=mcs
            )[0]
        )

        # observation rate gradients
        g[d, :] = -torch.einsum("ij->j", g[:d] * (1 - torch.eye(g.shape[1])))

        # regularization
        g[:d] += reg * torch.sign(theta[:d] * (1 - torch.eye(theta.shape[1])))
        g[d, :] += reg * torch.sign(theta[d, :])

        theta.grad = g

        print(f"{t} - {torch.linalg.norm(theta.grad)}")

        optimizer.step()

    with open(
        f"{results_filename[:-4]}_fold{k_index}{results_filename[-4:]}", "w"
    ) as f:
        for i in range(theta.shape[0]):
            f.write(" ".join(map(str, theta.numpy()[i])) + "\n")

    # get final score
    # create MHN theta matrix equivalent to current oMHN
    ctheta = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                ctheta[i, j] = theta[i, j]
            else:
                ctheta[i, j] = theta[i, j] - theta[d, j]

    curr_validation_score = fastmhn.approx.approx_gradient_and_score(
        ctheta, data_val, max_cluster_size=mcs
    )[1]
    average_validation_score += curr_validation_score

    print(f"Fold {k_index} finished, validation score: {curr_validation_score}")

average_validation_score /= k
print(f"Average validation score: {average_validation_score}")
