import numpy as np

import fastmhn

rng = np.random.default_rng(42)
np.random.seed(43)


# >>> setup


def get_clustering_with_seed(theta, seed=42, **kwargs):
    """Helper to get deterministic clustering by fixing random seed."""
    np.random.seed(seed)
    return fastmhn.clustering.hierarchical_clustering(theta, **kwargs)


# <<< setup


def test_clustering_returns_clusters():
    """Test that clustering returns valid clusters."""
    np.random.seed(43)
    d = 10
    theta = rng.normal(size=(d, d))
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, max_size=5, verbose=False
    )

    # Check return type
    assert isinstance(clustering, list), "Clustering should return a list"
    assert len(clustering) > 0, "Clustering should return non-empty list"

    # Check each cluster is valid
    for c in clustering:
        assert isinstance(
            c, list
        ), f"Each cluster should be a list, got {type(c)}"
        assert len(c) > 0, "Each cluster should be non-empty"
        assert all(
            isinstance(e, int) for e in c
        ), "Cluster elements should be integers"


def test_clustering_covers_all_events():
    """Test that all events are in exactly one cluster."""
    np.random.seed(43)
    d = 10
    theta = rng.normal(size=(d, d))
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, max_size=10, verbose=False
    )

    # Collect all events
    all_events = []
    for c in clustering:
        all_events.extend(c)

    # Check all events 0..d-1 are present
    assert sorted(all_events) == list(
        range(d)
    ), "All events should be in exactly one cluster"


def test_clustering_size_constraint():
    """Test that clusters respect max_size constraint."""
    np.random.seed(43)
    d = 20
    theta = rng.normal(size=(d, d))
    max_size = 5
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, max_size=max_size, verbose=False
    )

    for c in clustering:
        assert len(c) <= max_size, f"Cluster {c} has size {len(c)} > {max_size}"


def test_clustering_size_constraint_with_active_events():
    """Test that clusters respect max_size constraint with active_events."""
    np.random.seed(43)
    d = 10
    theta = rng.normal(size=(d, d))
    active_events = [0, 1, 2, 3, 4]
    max_size = 3
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta,
        e1=0,
        e2=1,
        max_size=max_size,
        active_events=active_events,
        verbose=False,
    )

    # Find cluster containing e1 and e2 (0 and 1)
    for c in clustering:
        if 0 in c and 1 in c:
            # Count active events in this cluster
            active_count = len([e for e in c if e in active_events])
            assert (
                active_count <= max_size
            ), f"Cluster {c} has {active_count} active events > {max_size}"
            break
    else:
        raise AssertionError("e1 and e2 should be in the same cluster")


def test_clustering_e1_e2_in_same_cluster():
    """Test that e1 and e2 end up in the same cluster."""
    np.random.seed(43)
    d = 10
    theta = rng.normal(size=(d, d))
    e1, e2 = 3, 7
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, e1=e1, e2=e2, max_size=8, verbose=False
    )

    # Find cluster containing e1
    cluster_with_e1 = None
    for c in clustering:
        if e1 in c:
            cluster_with_e1 = c
            break

    assert cluster_with_e1 is not None, f"Event {e1} not found in any cluster"
    assert (
        e2 in cluster_with_e1
    ), f"e1={e1} and e2={e2} should be in same cluster"


def test_clustering_deterministic_with_seed():
    """Test that clustering is deterministic when random seed is fixed."""
    np.random.seed(43)
    d = 8
    theta = rng.normal(size=(d, d))

    clustering1 = get_clustering_with_seed(theta, seed=123, max_size=5)
    clustering2 = get_clustering_with_seed(theta, seed=123, max_size=5)

    # Sort clusters for comparison
    def sort_clustering(clustering):
        return tuple(sorted(tuple(sorted(c)) for c in clustering))

    assert sort_clustering(clustering1) == sort_clustering(
        clustering2
    ), "Same seed should give same clustering"


def test_clustering_256_hard_limit():
    """Test that no cluster exceeds 256 elements."""
    np.random.seed(43)
    d = 300
    theta = rng.normal(size=(d, d))
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, max_size=d, verbose=False
    )

    for c in clustering:
        assert len(c) <= 256, f"Cluster size {len(c)} exceeds 256 limit"


def test_clustering_single_event():
    """Test edge case with single event."""
    np.random.seed(43)
    d = 1
    theta = np.array([[1.0]])
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, max_size=1, verbose=False
    )

    assert len(clustering) == 1, "Should have exactly one cluster"
    assert clustering[0] == [0], "Cluster should contain event 0"


def test_clustering_no_e1_e2():
    """Test clustering without specifying e1 and e2."""
    np.random.seed(43)
    d = 10
    theta = rng.normal(size=(d, d))
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, max_size=5, verbose=False
    )

    # Should still produce valid clustering
    assert len(clustering) > 0
    all_events = []
    for c in clustering:
        all_events.extend(c)
    assert sorted(all_events) == list(range(d))


def test_clustering_verbose_output():
    """Test that verbose mode doesn't crash."""
    np.random.seed(43)
    d = 5
    theta = rng.normal(size=(d, d))
    clustering = fastmhn.clustering.hierarchical_clustering(
        theta, max_size=3, verbose=True
    )

    # Just check it runs without error
    assert len(clustering) > 0


# >>> Phase 3: Property-based tests <<<


def test_clustering_preserves_all_events():
    """Test that clustering always covers all events exactly once."""
    np.random.seed(43)
    for d in range(1, 8):
        theta = rng.normal(size=(d, d))
        for max_size in [1, 2, d]:
            clustering = get_clustering_with_seed(theta, seed=42, max_size=max_size)
            
            # Flatten all clusters
            all_events = []
            for c in clustering:
                all_events.extend(c)
            
            # Each event 0..d-1 should appear exactly once
            assert sorted(all_events) == list(range(d)), (
                f"Events not preserved for d={d}, max_size={max_size}"
            )


def test_clustering_size_monotonic():
    """Test that smaller max_size produces more clusters."""
    np.random.seed(43)
    d = 10
    theta = rng.normal(size=(d, d))
    
    clustering_large = get_clustering_with_seed(theta, seed=42, max_size=d)
    clustering_small = get_clustering_with_seed(theta, seed=42, max_size=2)
    
    assert len(clustering_small) >= len(clustering_large), (
        f"Smaller max_size should produce at least as many clusters"
    )


def test_clustering_disjoint_clusters():
    """Test that clusters are disjoint (no overlapping events)."""
    np.random.seed(43)
    d = 10
    theta = rng.normal(size=(d, d))
    clustering = get_clustering_with_seed(theta, seed=42, max_size=5)
    
    # Check all pairs of clusters are disjoint
    for i, c1 in enumerate(clustering):
        for c2 in clustering[i + 1:]:
            assert len(set(c1) & set(c2)) == 0, (
                f"Clusters {c1} and {c2} have overlapping events"
            )
