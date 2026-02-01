"""
Post-training visualization: velocity fields, streamlines, and combined plots.

Run this after run_forward.py has trained a model.
For now, we re-train a quick model inline (later we'll add model checkpointing).
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.core.pinn import PINN
from src.core.sampling import sample_interior, sample_initial
from src.equations.omega_rans import make_residual_fn
from src.equations.stuart_base_flow import (
    stream_function, velocity, vorticity
)


# =============================================================================
# Re-train model (same config as run_forward.py)
# =============================================================================
def train_model():
    """Train model and return it. Later we'll load from checkpoint."""
    A, nu = 0.5, 0.01
    x_range = (0.0, 2 * float(jnp.pi))
    y_range = (-3.0, 3.0)
    t_range = (0.0, 1.0)

    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    model = PINN(in_dim=3, out_dim=1, hidden_dims=[32, 32, 32], key=subkey)

    def omega_tilde_ic(x, y):
        amplitude = 0.1
        sigma_x, sigma_y = 0.8, 0.8
        x0, y0 = jnp.pi, 0.0
        envelope = jnp.exp(-((x - x0)**2 / (2 * sigma_x**2)
                             + (y - y0)**2 / (2 * sigma_y**2)))
        return amplitude * y * envelope

    def loss_fn(model, batch):
        residual_fn = make_residual_fn(model, A=A, nu=nu)

        x_int = batch['interior']['x']
        y_int = batch['interior']['y']
        t_int = batch['interior']['t']

        def pde_residual_sq(x, y, t):
            return residual_fn(x, y, t)**2

        loss_pde = jnp.mean(jax.vmap(pde_residual_sq)(x_int, y_int, t_int))

        x_ic, y_ic, t_ic = batch['initial']['x'], batch['initial']['y'], batch['initial']['t']

        def ic_error_sq(x, y, t):
            xyt = jnp.stack([x, y, t])
            def psi_scalar(xyt_):
                return model(xyt_).squeeze()
            H = jax.hessian(psi_scalar)(xyt)
            wt = -(H[0, 0] + H[1, 1])
            return (wt - omega_tilde_ic(x, y))**2

        loss_ic = jnp.mean(jax.vmap(ic_error_sq)(x_ic, y_ic, t_ic))
        return loss_pde + 50.0 * loss_ic

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        return eqx.apply_updates(model, updates), new_opt_state, loss

    print("Training model (5000 steps)...")
    for i in range(5000):
        if i % 2000 == 0:
            key, k1, k2 = random.split(key, 3)
            batch = {
                'interior': sample_interior(k1, 500, x_range, y_range, t_range),
                'initial': sample_initial(k2, 200, x_range, y_range, t0=0.0),
            }
        model, opt_state, loss_val = step(model, opt_state, batch)
        if i % 1000 == 0:
            print(f"  Step {i:5d} | Loss: {loss_val:.4e}")

    print(f"  Final loss: {loss_val:.4e}\n")
    return model, A


# =============================================================================
# Evaluation helpers
# =============================================================================
def eval_psi_tilde_on_grid(model, x_range, y_range, t_val, nx, ny):
    """Evaluate psi_tilde on a grid."""
    x_lin = jnp.linspace(x_range[0], x_range[1], nx)
    y_lin = jnp.linspace(y_range[0], y_range[1], ny)

    def eval_point(x, y):
        return model(jnp.stack([x, y, jnp.array(t_val)])).squeeze()

    def eval_row(y_val):
        return jax.vmap(lambda x: eval_point(x, y_val))(x_lin)

    return jax.vmap(eval_row)(y_lin)


def eval_velocity_tilde_on_grid(model, x_range, y_range, t_val, nx, ny):
    """Evaluate v_tilde = (dpsi/dy, -dpsi/dx) on a grid."""
    x_lin = jnp.linspace(x_range[0], x_range[1], nx)
    y_lin = jnp.linspace(y_range[0], y_range[1], ny)

    def vel_at_point(x, y):
        xyt = jnp.stack([x, y, jnp.array(t_val)])
        g = jax.grad(lambda xyt_: model(xyt_).squeeze())(xyt)
        vt1 = g[1]    # dpsi/dy
        vt2 = -g[0]   # -dpsi/dx
        return vt1, vt2

    def eval_row(y_val):
        vt1s, vt2s = jax.vmap(lambda x: vel_at_point(x, y_val))(x_lin)
        return vt1s, vt2s

    results = jax.vmap(eval_row)(y_lin)
    return results[0], results[1]  # (ny, nx) each


def eval_omega_tilde_on_grid(model, x_range, y_range, t_val, nx, ny):
    """Evaluate omega_tilde = -laplacian(psi_tilde) on a grid."""
    x_lin = jnp.linspace(x_range[0], x_range[1], nx)
    y_lin = jnp.linspace(y_range[0], y_range[1], ny)

    def wt_at_point(x, y):
        xyt = jnp.stack([x, y, jnp.array(t_val)])
        H = jax.hessian(lambda xyt_: model(xyt_).squeeze())(xyt)
        return -(H[0, 0] + H[1, 1])

    def eval_row(y_val):
        return jax.vmap(lambda x: wt_at_point(x, y_val))(x_lin)

    return jax.vmap(eval_row)(y_lin)


# =============================================================================
# Plotting
# =============================================================================
def make_plots(model, A):
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    x_range = (0.0, 2 * float(jnp.pi))
    y_range = (-3.0, 3.0)
    nx, ny = 80, 80
    x_np = np.linspace(x_range[0], x_range[1], nx)
    y_np = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x_np, y_np)

    times = [0.0, 0.5, 1.0]

    # =========================================================================
    # Figure 1: Fluctuation velocity field (v_tilde) + omega_tilde contours
    # =========================================================================
    print("Computing velocity fields...")
    fig, axes = plt.subplots(1, len(times), figsize=(6 * len(times), 5))

    for idx, t_val in enumerate(times):
        ax = axes[idx]

        # omega_tilde as background color
        wt = np.array(eval_omega_tilde_on_grid(model, x_range, y_range, t_val, nx, ny))
        vmax = max(np.abs(wt).max(), 1e-8)
        ax.pcolormesh(X, Y, wt, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')

        # v_tilde as quiver
        nq = 20  # quiver resolution
        vt1, vt2 = eval_velocity_tilde_on_grid(model, x_range, y_range, t_val, nq, nq)
        vt1_np, vt2_np = np.array(vt1), np.array(vt2)
        xq = np.linspace(x_range[0], x_range[1], nq)
        yq = np.linspace(y_range[0], y_range[1], nq)
        Xq, Yq = np.meshgrid(xq, yq)
        speed = np.sqrt(vt1_np**2 + vt2_np**2)
        ax.quiver(Xq, Yq, vt1_np, vt2_np, color='k', alpha=0.6,
                  scale=speed.max() * 15 if speed.max() > 0 else 1)

        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')

    fig.suptitle(r"$\tilde{\omega}$ (color) + $\tilde{v}$ (arrows)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'velocity_field.png'), dpi=150, bbox_inches='tight')
    print("Saved velocity_field.png")

    # =========================================================================
    # Figure 2: Total streamlines (psi_base + psi_tilde) vs base streamlines
    # =========================================================================
    print("Computing streamlines...")
    fig, axes = plt.subplots(2, len(times), figsize=(6 * len(times), 8))

    for idx, t_val in enumerate(times):
        # Base stream function
        psi_base = np.array(jax.vmap(
            lambda y_: jax.vmap(lambda x_: stream_function(x_, y_, A))(jnp.array(x_np))
        )(jnp.array(y_np)))

        # Fluctuation stream function
        psi_tilde = np.array(eval_psi_tilde_on_grid(model, x_range, y_range, t_val, nx, ny))

        # Total
        psi_total = psi_base + psi_tilde

        # Top row: base streamlines only
        ax = axes[0, idx]
        n_levels = 20
        ax.contour(X, Y, psi_base, levels=n_levels, colors='k', linewidths=0.5)
        omega_base = np.array(jax.vmap(
            lambda y_: jax.vmap(lambda x_: vorticity(x_, y_, A))(jnp.array(x_np))
        )(jnp.array(y_np)))
        vmax_b = max(np.abs(omega_base).max(), 1e-8)
        ax.pcolormesh(X, Y, omega_base, cmap='RdBu_r', vmin=-vmax_b, vmax=vmax_b,
                      shading='auto', alpha=0.4)
        ax.set_title(f"Base flow (t={t_val:.2f})")
        ax.set_aspect('equal')

        # Bottom row: total streamlines
        ax = axes[1, idx]
        # Background: omega_base + omega_tilde
        wt = np.array(eval_omega_tilde_on_grid(model, x_range, y_range, t_val, nx, ny))
        omega_total = omega_base + wt
        vmax_t = max(np.abs(omega_total).max(), 1e-8)
        ax.pcolormesh(X, Y, omega_total, cmap='RdBu_r', vmin=-vmax_t, vmax=vmax_t,
                      shading='auto', alpha=0.4)
        ax.contour(X, Y, psi_total, levels=n_levels, colors='k', linewidths=0.5)
        ax.set_title(f"Base + perturbation (t={t_val:.2f})")
        ax.set_aspect('equal')

    axes[0, 0].set_ylabel("Base only")
    axes[1, 0].set_ylabel("Total (base + pert.)")
    fig.suptitle("Streamlines: base flow vs perturbed flow", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'streamlines.png'), dpi=150, bbox_inches='tight')
    print("Saved streamlines.png")

    # =========================================================================
    # Figure 3: Perturbation energy over time
    # =========================================================================
    print("Computing perturbation energy...")
    t_vals = np.linspace(0, 1, 20)
    energies = []
    for t_val in t_vals:
        wt = np.array(eval_omega_tilde_on_grid(model, x_range, y_range, float(t_val), nx, ny))
        # Enstrophy proxy: integral of omega_tilde^2
        dx = (x_range[1] - x_range[0]) / nx
        dy = (y_range[1] - y_range[0]) / ny
        enstrophy = np.sum(wt**2) * dx * dy
        energies.append(enstrophy)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_vals, energies, 'b-o', markersize=4)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\int \tilde{\omega}^2 \, dA$  (enstrophy)")
    ax.set_title(r"Fluctuation enstrophy $\|\tilde{\omega}\|^2$ over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'enstrophy.png'), dpi=150, bbox_inches='tight')
    print("Saved enstrophy.png")

    print(f"\nAll outputs in: {output_dir}/")


if __name__ == '__main__':
    model, A = train_model()
    make_plots(model, A)
