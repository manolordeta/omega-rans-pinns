"""
Collocation point sampling for PINN training.
"""

import jax
import jax.numpy as jnp


def sample_interior(key, n_points: int, x_range: tuple, y_range: tuple,
                    t_range: tuple) -> dict:
    """Sample random interior collocation points in (x, y, t) domain.

    Args:
        key: JAX PRNG key
        n_points: number of points to sample
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        t_range: (t_min, t_max)

    Returns:
        dict with 'x', 'y', 't' arrays of shape (n_points,)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (n_points,), minval=x_range[0], maxval=x_range[1])
    y = jax.random.uniform(k2, (n_points,), minval=y_range[0], maxval=y_range[1])
    t = jax.random.uniform(k3, (n_points,), minval=t_range[0], maxval=t_range[1])
    return {'x': x, 'y': y, 't': t}


def sample_initial(key, n_points: int, x_range: tuple, y_range: tuple,
                   t0: float = 0.0) -> dict:
    """Sample points on the initial condition surface t = t0."""
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (n_points,), minval=x_range[0], maxval=x_range[1])
    y = jax.random.uniform(k2, (n_points,), minval=y_range[0], maxval=y_range[1])
    t = jnp.full((n_points,), t0)
    return {'x': x, 'y': y, 't': t}


def sample_periodic_boundary(key, n_points: int, x_period: float,
                             y_range: tuple, t_range: tuple) -> dict:
    """Sample pairs of points on periodic boundaries x=0 and x=x_period.

    Returns dict with 'left' and 'right' sub-dicts for enforcing
    f(0, y, t) = f(x_period, y, t).
    """
    k1, k2 = jax.random.split(key)
    y = jax.random.uniform(k1, (n_points,), minval=y_range[0], maxval=y_range[1])
    t = jax.random.uniform(k2, (n_points,), minval=t_range[0], maxval=t_range[1])
    return {
        'left': {'x': jnp.zeros(n_points), 'y': y, 't': t},
        'right': {'x': jnp.full(n_points, x_period), 'y': y, 't': t},
    }


def make_batch_fn(n_interior: int, n_initial: int, n_boundary: int,
                  x_range: tuple, y_range: tuple, t_range: tuple,
                  x_period: float = 2 * jnp.pi):
    """Create a batch sampling function for omega-RANS training.

    Returns a function: key -> batch dict
    """
    def batch_fn(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            'interior': sample_interior(k1, n_interior, x_range, y_range, t_range),
            'initial': sample_initial(k2, n_initial, x_range, y_range, t0=t_range[0]),
            'boundary': sample_periodic_boundary(k3, n_boundary, x_period, y_range, t_range),
        }
    return batch_fn
