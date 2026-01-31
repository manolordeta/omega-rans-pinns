"""
Omega-RANS equation residuals for 2D incompressible flow.

Optimized version: computes all needed derivatives of psi_tilde in a single
forward/backward pass using jax.jacfwd, avoiding deeply nested jax.grad calls
that cause exponential JIT compile times.

The PINN outputs psi_tilde(x, y, t). From this we derive:
    v_tilde = (dpsi/dy, -dpsi/dx)
    w_tilde = -(d2psi/dx2 + d2psi/dy2)

The PDE residual is:
    R = dw_tilde/dt + (v . nabla)w_tilde + (v_tilde . nabla)w
        + (v_tilde . nabla)w_tilde - nu * laplacian(w_tilde)
        + F_base

where F_base = -nu * laplacian(omega_base) for Stuart base flow.
"""

import jax
import jax.numpy as jnp

from ..equations.stuart_base_flow import velocity, vorticity_gradients


def _laplacian_omega_base(x, y, A=0.5):
    """Analytic laplacian of Stuart base vorticity."""
    C = jnp.cos(x)
    S = jnp.sin(x)
    ch = jnp.cosh(y)
    sh = jnp.sinh(y)
    D = ch + A * C
    D4 = D**4
    coeff = 1.0 - A**2

    d2f_dx2 = 2.0 * A * (C * D + 3.0 * A * S**2) / D4
    d2f_dy2 = 2.0 * (-ch * D + 3.0 * sh**2) / D4

    return -coeff * (d2f_dx2 + d2f_dy2)


def make_residual_fn(model, A=0.5, nu=0.01):
    """Create residual function using efficient derivative computation.

    Instead of nesting jax.grad many times, we compute the full Hessian
    of psi_tilde w.r.t. (x, y, t) in one call, then extract what we need.

    Args:
        model: PINN callable, maps R^3 -> R^1
        A: Stuart parameter
        nu: viscosity

    Returns:
        residual_fn(x, y, t) -> scalar
    """

    def psi_scalar(xyt):
        """psi_tilde as a function of a single 3-vector."""
        return model(xyt).squeeze()

    # First derivatives: gradient of psi w.r.t. input
    grad_psi = jax.grad(psi_scalar)

    # Second derivatives: Hessian of psi w.r.t. input (3x3 matrix)
    hessian_psi = jax.jacfwd(grad_psi)

    def residual_at_point(x, y, t):
        xyt = jnp.stack([x, y, t])

        # --- First derivatives of psi_tilde ---
        g = grad_psi(xyt)  # [dpsi/dx, dpsi/dy, dpsi/dt]
        dpsi_dx, dpsi_dy, dpsi_dt = g[0], g[1], g[2]

        # Fluctuation velocity
        vt1 = dpsi_dy     # v_tilde_x = dpsi/dy
        vt2 = -dpsi_dx    # v_tilde_y = -dpsi/dx

        # --- Second derivatives of psi_tilde (Hessian) ---
        H = hessian_psi(xyt)  # 3x3 matrix
        # H[i,j] = d2psi / d(xyt_i) d(xyt_j)
        d2psi_dxdx = H[0, 0]
        d2psi_dydy = H[1, 1]
        d2psi_dxdt = H[0, 2]
        d2psi_dydt = H[1, 2]
        d2psi_dxdy = H[0, 1]

        # w_tilde = -(d2psi/dx2 + d2psi/dy2)
        wt = -(d2psi_dxdx + d2psi_dydy)

        # --- Derivatives of w_tilde ---
        # We need third derivatives of psi_tilde for grad(w_tilde) and
        # fourth derivatives for laplacian(w_tilde).
        # Use jacfwd of the hessian for third derivatives.

        # Third derivatives: Jacobian of Hessian w.r.t. input (3x3x3 tensor)
        jac_hessian = jax.jacfwd(hessian_psi)(xyt)  # shape (3,3,3)
        # jac_hessian[i,j,k] = d3psi / d(xyt_i) d(xyt_j) d(xyt_k)

        # dw_tilde/dt = -(d3psi/dx2dt + d3psi/dy2dt)
        dwt_dt = -(jac_hessian[0, 0, 2] + jac_hessian[1, 1, 2])

        # dw_tilde/dx = -(d3psi/dx3 + d3psi/dy2dx)
        dwt_dx = -(jac_hessian[0, 0, 0] + jac_hessian[1, 1, 0])

        # dw_tilde/dy = -(d3psi/dx2dy + d3psi/dy3)
        dwt_dy = -(jac_hessian[0, 0, 1] + jac_hessian[1, 1, 1])

        # For laplacian(w_tilde) we need fourth derivatives.
        # laplacian(w_tilde) = d2wt/dx2 + d2wt/dy2
        # = -(d4psi/dx4 + d4psi/dy2dx2) + -(d4psi/dx2dy2 + d4psi/dy4)
        # = -(d4psi/dx4 + 2*d4psi/dx2dy2 + d4psi/dy4)

        # Fourth derivatives via jacfwd of jac_hessian
        jac4 = jax.jacfwd(jax.jacfwd(hessian_psi))(xyt)  # shape (3,3,3,3)
        # jac4[i,j,k,l] = d4psi / d(xyt_i) d(xyt_j) d(xyt_k) d(xyt_l)

        d4psi_dx4 = jac4[0, 0, 0, 0]
        d4psi_dx2dy2 = jac4[0, 0, 1, 1]
        d4psi_dy4 = jac4[1, 1, 1, 1]

        lap_wt = -(d4psi_dx4 + 2.0 * d4psi_dx2dy2 + d4psi_dy4)

        # --- Base flow (analytic) ---
        v1, v2 = velocity(x, y, A)
        domega_dx, domega_dy = vorticity_gradients(x, y, A)

        # --- Assemble residual ---
        # dw_tilde/dt
        r = dwt_dt
        # + (v . nabla) w_tilde
        r = r + v1 * dwt_dx + v2 * dwt_dy
        # + (v_tilde . nabla) omega_base
        r = r + vt1 * domega_dx + vt2 * domega_dy
        # + (v_tilde . nabla) w_tilde  [nonlinear]
        r = r + vt1 * dwt_dx + vt2 * dwt_dy
        # - nu * laplacian(w_tilde)
        r = r - nu * lap_wt
        # + base forcing: -nu * laplacian(omega_base)
        r = r - nu * _laplacian_omega_base(x, y, A)

        return r

    return residual_at_point
