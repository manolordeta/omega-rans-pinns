"""
Core PINN module.

A Physics-Informed Neural Network parametrizes a solution field and is trained
by minimizing PDE residuals computed via automatic differentiation.
"""

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from typing import Sequence, Callable


class MLP(eqx.Module):
    """Multi-layer perceptron with configurable activation."""

    layers: list
    activation: Callable

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Sequence[int],
                 activation: Callable = jnp.tanh, *, key: jax.Array):
        keys = random.split(key, len(hidden_dims) + 1)
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class PINN(eqx.Module):
    """Physics-Informed Neural Network.

    Wraps an MLP and provides helper methods for computing derivatives
    of the network output with respect to its inputs via autodiff.
    """

    net: MLP

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Sequence[int],
                 activation: Callable = jnp.tanh, *, key: jax.Array):
        self.net = MLP(in_dim, out_dim, hidden_dims, activation, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)

    def predict(self, x: jax.Array) -> jax.Array:
        """Evaluate network on a single input point (unbatched)."""
        return self.net(x)

    def predict_batch(self, x: jax.Array) -> jax.Array:
        """Evaluate network on a batch of input points."""
        return jax.vmap(self.net)(x)
