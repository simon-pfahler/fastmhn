from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def hierarchical_clustering(
    theta: NDArray[np.float64],
    e1: Optional[int] = None,
    e2: Optional[int] = None,
    max_size: Optional[int] = None,
    active_events: Optional[list[int]] = None,
    verbose: bool = False,
) -> list[list[int]]:
    """
    Performs hierarchical clustering based on a theta matrix.

    The clustering respects constraints on cluster sizes. When e1 and e2 are
    specified, they are forced to be in the same cluster. When active_events
    is specified, the size constraint applies only to events in this list.

    Parameters
    ----------
    theta : numpy.ndarray
        dxd theta matrix used to determine cluster distances
    e1 : int, optional
        First event of an important event pair that must be in the same cluster.
        Default is None.
    e2 : int, optional
        Second event of an important event pair that must be in the same cluster.
        Default is None.
    max_size : int, optional
        Maximum allowed size for the cluster containing e1 and e2 (if specified),
        or for any cluster (if e1/e2 not specified). Default is None, which uses d.
    active_events : list, optional
        List of relevant events for size restriction. When specified, the
        max_size constraint applies only to events in this list. Default is None,
        which uses all events.
    verbose : bool, optional
        If True, prints clustering steps. Default is False.

    Returns
    -------
    list of list
        List of clusters, where each cluster is a list of event indices.
        All events 0..d-1 are included in exactly one cluster.
    """
    d = theta.shape[0]
    if max_size is None:
        max_size = d
    restrict_only_first_size = True
    if e1 is None and e2 is None:
        restrict_only_first_size = False
        e1 = 0
        e2 = 0
    clustering: list[list[int]] = [
        list({e1, e2}),
        *[[i] for i in range(d) if i not in (e1, e2)],
    ]

    # set active_events to the whole list if it is None
    if active_events is None:
        active_events = set(range(d))

    # fix order if it was broken by the set
    if clustering[0][0] != e1:
        clustering[0] = clustering[0][::-1]

    # we store inverse cluster distances because many distances are infinity
    inv_cluster_distances = np.zeros((len(clustering), len(clustering)))
    # add tiny random noise so we get clusters of size >1 even when theta is
    # diagonal
    inv_cluster_distances += np.random.normal(
        loc=0,
        scale=1e-5,
        size=inv_cluster_distances.shape,
    )
    # Set diagonal to 0 to avoid self-combination
    np.fill_diagonal(inv_cluster_distances, 0)
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
        combine_clusters: list[int] = sorted(
            np.unravel_index(
                inv_cluster_distances.argmax(), inv_cluster_distances.shape
            )
        )

        # check whether the combination is allowed
        allowed = True
        if restrict_only_first_size:
            if combine_clusters[0] == 0:
                allowed = (
                    len(
                        [
                            e
                            for e in clustering[combine_clusters[0]]
                            if e in active_events
                        ]
                    )
                    + len(
                        [
                            e
                            for e in clustering[combine_clusters[1]]
                            if e in active_events
                        ]
                    )
                    <= max_size
                )
        else:
            allowed = (
                len(
                    [
                        e
                        for e in clustering[combine_clusters[0]]
                        if e in active_events
                    ]
                )
                + len(
                    [
                        e
                        for e in clustering[combine_clusters[1]]
                        if e in active_events
                    ]
                )
                <= max_size
            )
        if (
            len(clustering[combine_clusters[0]])
            + len(clustering[combine_clusters[1]])
            > 256
        ):
            allowed = False

        # set corresponding distance to infinity
        if not allowed:
            inv_cluster_distances[combine_clusters[0], combine_clusters[1]] = 0
            inv_cluster_distances[combine_clusters[1], combine_clusters[0]] = 0
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
