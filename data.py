
import numpy as np
def get_two_gaussian_clusters(num_points = 150,
                              noise = 2,
                              c1_center = [-3, -3],
                              c2_center = [3, 3]):
    np.random.seed(42)
    # Parameters
    # inputs
    np_c1_center = np.array(c1_center)
    cluster_1 = np.random.normal(
        loc=np_c1_center, scale=noise, size=(num_points, 2)
    )

    np_c2_center = np.array(c2_center)
    cluster_2 = np.random.normal(
        loc=np_c2_center, scale=noise, size=(num_points, 2)
    )
    return combine_and_shuffle(cluster_1, cluster_2)

def combine_and_shuffle(cluster_1, cluster_2, seed=None):
    rng = np.random.default_rng(seed)

    # 1. Combine points along the first axis
    combined_points = np.concatenate([cluster_1, cluster_2], axis=0)

    # 2. Create label array: 1.0 for cluster_1, 0.0 for cluster_2
    labels = np.concatenate(
        [
            np.ones(len(cluster_1), dtype=np.float64),
            np.zeros(len(cluster_2), dtype=np.float64),
        ]
    )

    # 3. Generate a shared permutation and shuffle in unison
    indices = rng.permutation(len(combined_points))

    return combined_points[indices], labels[indices]