import numpy as np

import fastmhn

results_filename = "theta.dat"

d = 3
N = 100
data = np.random.randint(2, size=(N, d), dtype=np.int32)

# level of cross-validation
k = 5

# >>> optimization parameters
gradient_and_score_params = {"max_cluster_size": 10}
reg = 1e-2
adam_params = {
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

# >>> shuffle data
rng = np.random.default_rng(42)
shuffled_indices = np.arange(data.shape[0])
rng.shuffle(shuffled_indices)
data = data[shuffled_indices, :]
# <<< shuffle data

# >>> get fold sizes for cross-validation
fold_sizes = (N // k) * np.ones(k, dtype=int)
fold_sizes[: N % k] += 1
# <<< get fold sizes for cross-validation

score_offset = fastmhn.utility.get_score_offset(data)
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

    theta = fastmhn.learn.learn_omhn(
        data_train,
        reg=reg,
        gradient_and_score_params=gradient_and_score_params,
        adam_params=adam_params,
    )

    # get final score
    ctheta = fastmhn.utility.cmhn_from_omhn(theta)

    curr_validation_score = fastmhn.approx.approx_gradient_and_score(
        ctheta, data_val, **gradient_and_score_params
    )[1]
    average_validation_score += curr_validation_score

    print(
        f"Fold {k_index} finished, validation score: "
        f"{curr_validation_score} (offset {score_offset})"
    )

average_validation_score /= k
print(
    f"Average validation score for reg {reg:.0e}: "
    f"{average_validation_score} (offset {score_offset})"
)
