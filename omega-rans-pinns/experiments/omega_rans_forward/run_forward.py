"""
First experiment: solve omega-RANS forward problem.

Given:
    - Base flow: Stuart cats-eye (v, omega known analytically)
    - Initial condition: omega_tilde(x, y, 0) = localized perturbation
    - Boundary conditions: periodic in x, decay in y

Find: omega_tilde(x, y, t) for t in [0, T]

The PINN parametrizes psi_tilde(x, y, t) (fluctuation stream function).
From psi_tilde we derive v_tilde and omega_tilde via autodiff.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np

from src.core.pinn import PINN
from src.core.sampling import sample_interior, sample_initial
from src.equations.omega_rans import make_residual_fn, compute_base_forcing
from src.equations.stuart_base_flow import vorticity
from src.utils.visualization import plot_loss_history, plot_field_2d, evaluate_on_grid


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Stuart cats-eye parameter
    'A': 0.5,
    # Viscosity
    'nu': 0.01,
    # Domain
    'x_range': (0.0, 2 * jnp.pi),
    'y_range': (-4.0, 4.0),
    't_range': (0.0, 1.0),
    # Network architecture
    'hidden_dims': [64, 64, 64, 64],
    # Training
    'n_steps': 20000,
    'lr': 1e-3,
    'n_interior': 4000,
    'n_initial': 1000,
    # Loss weights
    'w_pde': 1.0,
    'w_ic': 50.0,
    # Logging
    'log_every': 1000,
    'seed': 42,
}


# =============================================================================
# Initial condition for omega_tilde
# =============================================================================
def omega_tilde_ic(x, y, A=0.5):
    """Initial perturbation: localized vortex dipole near a saddle point.

    A Gaussian-modulated sinusoidal perturbation centered near (pi, 0),
    which is the hyperbolic stagnation point of Stuart cats-eye.
    This is where we'd expect the most interesting interaction.
    """
    amplitude = 0.1
    sigma_x = 0.8
    sigma_y = 0.8
    x0, y0 = jnp.pi, 0.0
    envelope = jnp.exp(-((x - x0)**2 / (2 * sigma_x**2)
                         + (y - y0)**2 / (2 * sigma_y**2)))
    # Dipole structure (odd in y around y0)
    return amplitude * y * envelope


# To enforce this IC, we need psi_tilde such that -laplacian(psi_tilde) = omega_tilde_ic.
# Instead, we softly enforce that the resulting omega_tilde at t=0 matches the IC.


# =============================================================================
# Loss function
# =============================================================================
def make_loss_fn(A, nu, w_pde, w_ic):
    """Build the total loss function."""

    def loss_fn(model, batch):
        residual_fn = make_residual_fn(model, A=A, nu=nu)

        # --- PDE residual loss (interior) ---
        x_int = batch['interior']['x']
        y_int = batch['interior']['y']
        t_int = batch['interior']['t']

        def pde_residual_sq(x, y, t):
            r = residual_fn(x, y, t)
            return r**2

        pde_residuals = jax.vmap(pde_residual_sq)(x_int, y_int, t_int)
        loss_pde = jnp.mean(pde_residuals)

        # --- Initial condition loss ---
        # omega_tilde = -laplacian(psi_tilde) should match omega_tilde_ic at t=0
        x_ic = batch['initial']['x']
        y_ic = batch['initial']['y']
        t_ic = batch['initial']['t']

        def ic_error_sq(x, y, t):
            inp = jnp.stack([x, y, t])
            # Compute omega_tilde = -laplacian(psi_tilde)
            def psi_fn(x_, y_):
                return model(jnp.stack([x_, y_, t])).squeeze()
            d2x = jax.grad(jax.grad(psi_fn, 0), 0)(x, y)
            d2y = jax.grad(jax.grad(psi_fn, 1), 1)(x, y)
            wt = -(d2x + d2y)
            target = omega_tilde_ic(x, y, A)
            return (wt - target)**2

        ic_errors = jax.vmap(ic_error_sq)(x_ic, y_ic, t_ic)
        loss_ic = jnp.mean(ic_errors)

        return w_pde * loss_pde + w_ic * loss_ic

    return loss_fn


# =============================================================================
# Main
# =============================================================================
def main():
    cfg = CONFIG
    key = random.PRNGKey(cfg['seed'])

    # --- Create model ---
    key, subkey = random.split(key)
    model = PINN(
        in_dim=3,       # (x, y, t)
        out_dim=1,      # psi_tilde
        hidden_dims=cfg['hidden_dims'],
        key=subkey,
    )

    # --- Build loss ---
    loss_fn = make_loss_fn(cfg['A'], cfg['nu'], cfg['w_pde'], cfg['w_ic'])

    # --- Training ---
    optimizer = optax.adam(cfg['lr'])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    loss_history = []
    print("Starting training...")
    print(f"  Architecture: {cfg['hidden_dims']}")
    print(f"  Stuart A={cfg['A']}, nu={cfg['nu']}")
    print(f"  Steps: {cfg['n_steps']}, LR: {cfg['lr']}")
    print()

    for i in range(cfg['n_steps']):
        # Resample collocation points periodically
        if i % 2000 == 0:
            key, k1, k2 = random.split(key, 3)
            batch = {
                'interior': sample_interior(
                    k1, cfg['n_interior'],
                    cfg['x_range'], cfg['y_range'], cfg['t_range']),
                'initial': sample_initial(
                    k2, cfg['n_initial'],
                    cfg['x_range'], cfg['y_range'], t0=0.0),
            }

        model, opt_state, loss_val = step(model, opt_state, batch)
        loss_history.append(float(loss_val))

        if i % cfg['log_every'] == 0:
            print(f"Step {i:6d} | Loss: {loss_val:.6e}")

    print(f"\nFinal loss: {loss_history[-1]:.6e}")

    # --- Save outputs ---
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Loss curve
    plot_loss_history(loss_history, save_path=os.path.join(output_dir, 'loss.png'))

    # Evaluate omega_tilde at different times
    def omega_tilde_eval(x, y, t):
        """Compute omega_tilde = -laplacian(psi_tilde) at a point."""
        def psi_fn(x_, y_):
            return model(jnp.stack([x_, y_, t])).squeeze()
        d2x = jax.grad(jax.grad(psi_fn, 0), 0)(x, y)
        d2y = jax.grad(jax.grad(psi_fn, 1), 1)(x, y)
        return -(d2x + d2y)

    x_range = cfg['x_range']
    y_range = cfg['y_range']
    times = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
    for idx, t_val in enumerate(times):
        field = evaluate_on_grid(omega_tilde_eval, x_range, y_range,
                                 t_val, nx=80, ny=80)
        field_np = np.array(field)
        ax = axes[idx]
        vmax = max(np.abs(field_np).max(), 1e-8)
        x_lin = np.linspace(x_range[0], x_range[1], field_np.shape[1])
        y_lin = np.linspace(y_range[0], y_range[1], field_np.shape[0])
        X, Y = np.meshgrid(x_lin, y_lin)
        c = ax.pcolormesh(X, Y, field_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          shading='auto')
        fig.colorbar(c, ax=ax, fraction=0.046)
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_aspect('equal')

    fig.suptitle(r"$\tilde{\omega}(x, y, t)$ — omega-RANS forward solve", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'omega_tilde_evolution.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to {output_dir}/")

    # Also plot base flow vorticity for reference
    def base_omega(x, y, t):
        return vorticity(x, y, cfg['A'])

    base_field = evaluate_on_grid(base_omega, x_range, y_range, 0.0, nx=80, ny=80)
    plot_field_2d(np.array(base_field), x_range, y_range,
                  title=r"Base flow $\omega$ (Stuart cats-eye, A=0.5)",
                  save_path=os.path.join(output_dir, 'base_vorticity.png'))

    plt.show()


if __name__ == '__main__':
    main()
