
import numpy as np
def get_two_gaussian_clusters(n_points:int = 150,
                              noise: float = 2.0,
                              c1_center: tuple[float, float] = [-3, -3],
                              c2_center: tuple[float, float] = [3, 3]):
    np.random.seed(42)
    # Parameters
    # inputs
    np_c1_center = np.array(c1_center)
    cluster_1 = np.random.normal(
        loc=np_c1_center, scale=noise, size=(int(n_points/2), 2)
    )

    np_c2_center = np.array(c2_center)
    cluster_2 = np.random.normal(
        loc=np_c2_center, scale=noise, size=(int(n_points/2), 2)
    )
    return combine_and_shuffle(cluster_1, cluster_2)


def get_two_gaussian_circles(n_points:int = 150,
                              noise: float = 0.01,
                              c1_radius: float = 1.0,
                              c2_radius: float = 6.0):
    cluster_1 = generate_noisy_circular_points(n_points = int(n_points/2),
                                               radius = c1_radius,
                                               noise_std = noise)
    cluster_2 = generate_noisy_circular_points(n_points = int(n_points/2),
                                               radius = c2_radius,
                                               noise_std = noise)
    return combine_and_shuffle(cluster_1, cluster_2)

def generate_noisy_circular_points(
    n_points: int,
    radius: float = 1.0,
    noise_std: float = 0.05,
    center: tuple[float, float] = (0.0, 0.0),
    seed: int | None = None,
) -> np.ndarray:
    """Generate uniform points inside a 2D circle with additive Gaussian noise.

    Parameters:
        n_points: Number of points to sample.
        radius: Radius of the bounding circle.
        noise_std: Standard deviation of Gaussian noise added to coordinates.
        center: (x, y) coordinates for circle center.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (n_points, 2).
    """
    rng = np.random.default_rng(seed)

    # Uniform area sampling in polar coordinates:
    # Sampling r as sqrt(U) ensures uniform density across the disc area
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)

    # Convert polar to Cartesian coordinates
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    points = np.column_stack((x, y))

    # Add Gaussian noise
    noise = rng.normal(loc=0.0, scale=noise_std, size=points.shape)

    return points + noise

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