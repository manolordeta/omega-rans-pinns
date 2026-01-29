"""Visualization utilities for omega-RANS PINN experiments."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(loss_history, title="Training Loss", save_path=None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_field_2d(field_values, x_range, y_range, title="", save_path=None,
                  cmap='RdBu_r', nx=100, ny=100):
    """Plot a 2D scalar field as a filled contour.

    Args:
        field_values: (ny, nx) array
        x_range, y_range: (min, max) tuples
        title: plot title
        save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x_lin = np.linspace(x_range[0], x_range[1], field_values.shape[1])
    y_lin = np.linspace(y_range[0], y_range[1], field_values.shape[0])
    X, Y = np.meshgrid(x_lin, y_lin)
    vmax = np.abs(field_values).max()
    c = ax.pcolormesh(X, Y, field_values, cmap=cmap, vmin=-vmax, vmax=vmax,
                      shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect('equal')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def evaluate_on_grid(fn, x_range, y_range, t_val, nx=100, ny=100):
    """Evaluate a scalar function fn(x, y, t) on a 2D grid at fixed t.

    Args:
        fn: callable (x, y, t) -> scalar
        x_range, y_range: (min, max) tuples
        t_val: fixed time value
        nx, ny: grid resolution

    Returns:
        (ny, nx) array of function values
    """
    x_lin = jnp.linspace(x_range[0], x_range[1], nx)
    y_lin = jnp.linspace(y_range[0], y_range[1], ny)

    def eval_row(y_val):
        def eval_point(x_val):
            return fn(x_val, y_val, jnp.array(t_val))
        return jax.vmap(eval_point)(x_lin)

    return jax.vmap(eval_row)(y_lin)
