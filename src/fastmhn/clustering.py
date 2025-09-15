import numpy as np


def hierarchical_clustering(
    theta, e1=None, e2=None, max_size=None, verbose=False
):
    """
    Performs hierarchical clustering based on a `theta` matrix, with the
    restriction that `e1` and `e2` have to be in the same cluster, and that
    their cluster contains at most `max_size` events.
    When `e1` and `e2` are not specified, the algorithm makes sure that none of
    the clusters contain more than `max_size` events.

    `theta`: dxd theta matrix
    `e1`: first event of the important event pair, default is `None`
    `e2`: second event of the important event pair, default is `None`
    `max_size`: maximum allowed size for the cluster containing `e1` and `e2`
    `verbose`: set to `True` to get more output
    """

    d = theta.shape[0]
    if max_size == None:
        max_size = d
    restrict_only_first_size = True
    if e1 is None and e2 is None:
        restrict_only_first_size = False
        e1 = 0
        e2 = 0
    clustering = [
        list({e1, e2}),
        *[[i] for i in range(d) if i not in (e1, e2)],
    ]

    # fix order if it was broken by the set
    if clustering[0][0] != e1:
        clustering[0] = clustering[0][::-1]

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
        if restrict_only_first_size:
            if combine_clusters[0] == 0:
                if (
                    len(clustering[combine_clusters[0]])
                    + len(clustering[combine_clusters[1]])
                    > max_size
                ):
                    allowed = False
        else:
            allowed = (
                len(clustering[combine_clusters[0]])
                + len(clustering[combine_clusters[1]])
                <= max_size
            )

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
