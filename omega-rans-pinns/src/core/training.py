"""
Training loop for PINNs.

Supports Adam optimizer as baseline. Gauss-Newton / kfac-jax integration
will be added in src/optimizers/ for high-precision work.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Callable, NamedTuple


class TrainState(NamedTuple):
    step: int
    opt_state: optax.OptState
    best_loss: float


def make_train_step(loss_fn: Callable, optimizer: optax.GradientTransformation):
    """Create a JIT-compiled training step.

    Args:
        loss_fn: function(model, batch) -> scalar loss
        optimizer: optax optimizer

    Returns:
        step function(model, state, batch) -> (model, state, loss_value)
    """

    @eqx.filter_jit
    def step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    return step


def train(model, loss_fn, batch_fn, n_steps: int, lr: float = 1e-3,
          log_every: int = 500, resample_every: int = 5000,
          key=None):
    """Basic training loop.

    Args:
        model: equinox module (PINN)
        loss_fn: function(model, batch) -> scalar
        batch_fn: function(key) -> batch dict
        n_steps: number of training steps
        lr: learning rate for Adam
        log_every: print loss every N steps
        resample_every: resample collocation points every N steps
        key: JAX PRNG key

    Returns:
        (trained_model, loss_history)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    step_fn = make_train_step(loss_fn, optimizer)

    loss_history = []
    key, subkey = jax.random.split(key)
    batch = batch_fn(subkey)

    for i in range(n_steps):
        if i > 0 and i % resample_every == 0:
            key, subkey = jax.random.split(key)
            batch = batch_fn(subkey)

        model, opt_state, loss_val = step_fn(model, opt_state, batch)
        loss_history.append(float(loss_val))

        if i % log_every == 0:
            print(f"Step {i:6d} | Loss: {loss_val:.6e}")

    return model, loss_history
