import numpy as np

import fastmhn

torch.set_grad_enabled(False)

results_filename = "theta.dat"

d = 3
N = 100
data = np.random.randint(2, size=(N, d), dtype=np.int32)

# >>> optimization parameters
gradient_and_score_params = {"max_cluster_size": 10}
reg = 1e-2
adamW_params = {
    "alpha": 0.1,
    "beta1": 0.7,
    "beta2": 0.9,
    "eps": 1e-8,
    "verbose": True,
}
# <<< optimization parameters

# >>> Print information about dataset
avg_MB = np.mean(np.sum(data, axis=1))
max_MB = np.max(np.sum(data, axis=1))
nr_samples_approx = np.sum(
    np.sum(data, axis=1) > gradient_and_score_params["max_cluster_size"]
)
print(
    f"Dataset information:\n"
    f"\t{data.shape[0]} Patients\n"
    f"\tAverage mutational burden: {avg_MB}\n"
    f"\tMaximum mutational burden: {max_MB}\n"
    f"\tNumber of samples with MB > {gradient_and_score_params['max_cluster_size']}: {nr_samples_approx}"
)
# <<< Print information about dataset

theta = fastmhn.learn.learn_mhn(
    data,
    reg=reg,
    gradient_and_score_params=gradient_and_score_params,
    adamW_params=adamW_params,
)

np.savetxt(results_filename, theta)
