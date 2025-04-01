def get_clusters(theta, e1, e2, max_size):
    """
    Performs a clustering based on a `theta` matrix, with the restriction that
    `e1` and `e2` have to be in the same cluster, and that their cluster
    contains at most `max_size` events.

    `theta`: dxd theta matrix
    `e1`: first event of the important event pair
    `e2`: second event of the important event pair
    `max_size`: maximum allowed size for the cluster containing `e1` and `e2`
    """

    A = np.where(
        np.abs(theta) > np.abs(theta.T), np.abs(theta), np.abs(theta.T)
    )

    D = np.diag(np.sum(A, axis=0))

    L = D - A

    Dinvsqrt = np.diag(np.diag(D ** (-0.5)))

    L = Dinvsqrt @ L @ Dinvsqrt

    # TODO: Add a clustering algorithm here!
    return [(e1, e2), (i for i in range(theta.shape[0]) if i not in (e1, e2))]
