"""
Omega-RANS equation residuals for 2D incompressible flow.

From the paper (Eq. 4):
    dw/dt + dw_tilde/dt + (v . nabla)w + (v_tilde . nabla)w_tilde
    + (v . nabla)w_tilde + (v_tilde . nabla)w - nu * laplacian(w + w_tilde) = 0

Strategy:
    - The base flow v is known (Stuart cats-eye), so w = curl(v) is known and steady.
    - The PINN parametrizes psi_tilde(x, y, t): the fluctuation stream function.
    - From psi_tilde we derive:
        v_tilde = (d(psi_tilde)/dy, -d(psi_tilde)/dx)   [divergence-free by construction]
        w_tilde = -laplacian(psi_tilde)
    - All derivatives computed via JAX autodiff.

Since v, w are steady (Stuart):
    dw/dt = 0
    (v . nabla)w  is a known forcing term (computed analytically or via autodiff)
    -nu * laplacian(w) is a known forcing term

The PDE residual for the PINN becomes:
    R = dw_tilde/dt + (v . nabla)w_tilde + (v_tilde . nabla)w
        + (v_tilde . nabla)w_tilde - nu * laplacian(w_tilde)
        + F_base

where F_base = (v . nabla)w - nu * laplacian(w) collects the known base-flow terms.
"""

import jax
import jax.numpy as jnp
from functools import partial

from ..equations.stuart_base_flow import (
    velocity, vorticity, velocity_gradients, vorticity_gradients,
)


def _laplacian_omega(x, y, A=0.5, nu=0.01):
    """Analytic laplacian of Stuart vorticity: laplacian(omega).

    omega = -(1-A^2) / D^2,  D = cosh(y) + A*cos(x)

    d2(omega)/dx2 + d2(omega)/dy2.  Computed via chain rule.
    """
    C = jnp.cos(x)
    S = jnp.sin(x)
    ch = jnp.cosh(y)
    sh = jnp.sinh(y)
    D = ch + A * C
    coeff = 1.0 - A**2

    # Second derivatives of omega = -coeff / D^2
    # d/dx(omega) = 2*coeff*A*S / D^3
    # d2/dx2(omega) = 2*coeff*A*(C*D^3 - 3*D^2*(-A*S)*A*S) / D^6  ... simplify:
    # d2/dx2 = 2*coeff*A*(C*D + 3*A*S^2) / D^4  ... wait let me be more careful
    #
    # Let f = 1/D^2, then omega = -coeff * f
    # df/dx = -2/D^3 * dD/dx = -2/D^3 * (-A*S) = 2*A*S / D^3
    # d2f/dx2 = d/dx(2*A*S / D^3)
    #         = 2*A*C / D^3 - 2*A*S * 3/D^4 * (-A*S)
    #         = 2*A*C / D^3 + 6*A^2*S^2 / D^4
    #         = 2*A*(C*D + 3*A*S^2) / D^4
    #
    # df/dy = -2/D^3 * dD/dy = -2*sh / D^3
    # d2f/dy2 = d/dy(-2*sh / D^3)
    #         = -2*ch / D^3 - (-2*sh) * 3/D^4 * sh
    #         = -2*ch / D^3 + 6*sh^2 / D^4
    #         = (-2*ch*D + 6*sh^2) / D^4
    #         = 2*(-ch*D + 3*sh^2) / D^4

    D4 = D**4
    d2f_dx2 = 2.0 * A * (C * D + 3.0 * A * S**2) / D4
    d2f_dy2 = 2.0 * (-ch * D + 3.0 * sh**2) / D4

    return -coeff * (d2f_dx2 + d2f_dy2)


def compute_base_forcing(x, y, A=0.5, nu=0.01):
    """Compute F_base = (v . nabla)omega - nu * laplacian(omega).

    Since Stuart is a steady Euler solution, (v . nabla)omega = 0 for inviscid case.
    Actually, for Stuart cats-eye: omega = f(psi) = -(1-A^2)*exp(-2*psi), so
    (v . nabla)omega = omega'(psi) * (v . nabla)psi = 0 because v is tangent to
    streamlines, i.e., (v . nabla)psi = 0.

    So F_base = -nu * laplacian(omega).
    """
    lap_omega = _laplacian_omega(x, y, A=A, nu=nu)
    return -nu * lap_omega


def make_residual_fn(pinn_model, A=0.5, nu=0.01):
    """Create a function that computes the omega-RANS PDE residual.

    The PINN outputs psi_tilde(x, y, t) -- the fluctuation stream function.

    Args:
        pinn_model: callable (x, y, t) -> psi_tilde (scalar)
        A: Stuart parameter
        nu: kinematic viscosity

    Returns:
        residual_fn(x, y, t) -> scalar residual
    """

    def psi_tilde_fn(x, y, t):
        """Evaluate psi_tilde from the PINN. Input/output are scalars."""
        inp = jnp.stack([x, y, t])
        return pinn_model(inp).squeeze()

    def residual_at_point(x, y, t):
        """Compute PDE residual at a single (x, y, t) point."""

        # --- Fluctuation quantities from psi_tilde via autodiff ---
        # First derivatives of psi_tilde
        dpsi_dx = jax.grad(psi_tilde_fn, argnums=0)(x, y, t)
        dpsi_dy = jax.grad(psi_tilde_fn, argnums=1)(x, y, t)
        dpsi_dt = jax.grad(psi_tilde_fn, argnums=2)(x, y, t)

        # Fluctuation velocity: v_tilde = (dpsi/dy, -dpsi/dx)
        vt1 = dpsi_dy    # v_tilde_1
        vt2 = -dpsi_dx   # v_tilde_2

        # Second derivatives for laplacian(psi_tilde) and w_tilde
        d2psi_dx2 = jax.grad(jax.grad(psi_tilde_fn, 0), 0)(x, y, t)
        d2psi_dy2 = jax.grad(jax.grad(psi_tilde_fn, 1), 1)(x, y, t)

        # w_tilde = -laplacian(psi_tilde)
        wt = -(d2psi_dx2 + d2psi_dy2)

        # --- Derivatives of w_tilde ---
        # We need dw_tilde/dt, dw_tilde/dx, dw_tilde/dy, laplacian(w_tilde)
        # Define w_tilde as a function for further autodiff
        def wt_fn(x_, y_, t_):
            d2x = jax.grad(jax.grad(psi_tilde_fn, 0), 0)(x_, y_, t_)
            d2y = jax.grad(jax.grad(psi_tilde_fn, 1), 1)(x_, y_, t_)
            return -(d2x + d2y)

        dwt_dt = jax.grad(wt_fn, argnums=2)(x, y, t)
        dwt_dx = jax.grad(wt_fn, argnums=0)(x, y, t)
        dwt_dy = jax.grad(wt_fn, argnums=1)(x, y, t)

        # Laplacian of w_tilde
        d2wt_dx2 = jax.grad(jax.grad(wt_fn, 0), 0)(x, y, t)
        d2wt_dy2 = jax.grad(jax.grad(wt_fn, 1), 1)(x, y, t)
        lap_wt = d2wt_dx2 + d2wt_dy2

        # --- Base flow quantities (analytic, steady) ---
        v1, v2 = velocity(x, y, A)
        domega_dx, domega_dy = vorticity_gradients(x, y, A)

        # --- Assemble PDE residual ---
        # dw_tilde/dt
        term1 = dwt_dt

        # (v . nabla) w_tilde  [base velocity advects fluctuation vorticity]
        term2 = v1 * dwt_dx + v2 * dwt_dy

        # (v_tilde . nabla) w  [fluctuation velocity advects base vorticity]
        term3 = vt1 * domega_dx + vt2 * domega_dy

        # (v_tilde . nabla) w_tilde  [nonlinear self-interaction of fluctuation]
        term4 = vt1 * dwt_dx + vt2 * dwt_dy

        # -nu * laplacian(w_tilde)
        term5 = -nu * lap_wt

        # Base forcing: -nu * laplacian(omega)  [known, from viscous diffusion of base]
        f_base = compute_base_forcing(x, y, A=A, nu=nu)

        return term1 + term2 + term3 + term4 + term5 + f_base

    return residual_at_point
