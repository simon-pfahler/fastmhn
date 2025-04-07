import numpy as np


def get_clusters(theta, e1, e2, max_size=None, verbose=False):
    """
    Performs a clustering based on a `theta` matrix, with the restriction that
    `e1` and `e2` have to be in the same cluster, and that their cluster
    contains at most `max_size` events.

    `theta`: dxd theta matrix
    `e1`: first event of the important event pair
    `e2`: second event of the important event pair
    `max_size`: maximum allowed size for the cluster containing `e1` and `e2`
    `verbose`: set to `True` to get more output
    """

    d = theta.shape[0]
    if max_size == None:
        max_size = d
    clustering = [
        list({e1, e2}),
        *[[i] for i in range(d) if i not in (e1, e2)],
    ]

    # we store inverse cluster distances because many distances are infinity
    inv_cluster_distances = np.zeros((len(clustering), len(clustering)))
    for i in range(len(clustering)):
        for j in range(i):
            inv_cluster_distances[i, j] = max(
                max(np.abs(theta[e1, e2]), np.abs(theta[e2, e1]))
                for e1 in clustering[i]
                for e2 in clustering[j]
            )
    inv_cluster_distances += inv_cluster_distances.T

    if verbose:
        print("Clustering in process, starting point:")
        print(clustering)

    # perform clustering steps (unite two clusters) as long as possible
    while np.max(inv_cluster_distances) > 0:
        # clusters to combine next
        combine_clusters = sorted(
            np.unravel_index(
                inv_cluster_distances.argmax(), inv_cluster_distances.shape
            )
        )

        # check whether the combination is allowed
        allowed = True
        if combine_clusters[0] == 0:
            if (
                len(clustering[combine_clusters[0]])
                + len(clustering[combine_clusters[1]])
                > max_size
            ):
                allowed = False

        # set corresponding distance to infinity
        if not allowed:
            inv_cluster_distances[combine_clusters] = 0
            inv_cluster_distances[combine_clusters[::-1]] = 0
            continue

        # update the distance matrix
        inv_cluster_distances[combine_clusters[0]] = np.where(
            inv_cluster_distances[combine_clusters[0]]
            > inv_cluster_distances[combine_clusters[1]],
            inv_cluster_distances[combine_clusters[0]],
            inv_cluster_distances[combine_clusters[1]],
        )
        inv_cluster_distances[:, combine_clusters[0]] = inv_cluster_distances[
            combine_clusters[0]
        ]
        inv_cluster_distances[combine_clusters[0], combine_clusters[0]] = 0
        inv_cluster_distances = np.delete(
            np.delete(inv_cluster_distances, combine_clusters[1], axis=0),
            combine_clusters[1],
            axis=1,
        )

        # update the clustering
        clustering[combine_clusters[0]] += clustering[combine_clusters[1]]
        del clustering[combine_clusters[1]]

        if verbose:
            print("Clustering step taken:")
            print(clustering)

    return clustering
