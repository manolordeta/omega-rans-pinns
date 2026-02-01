"""
Stuart cats-eye vortex: exact stationary solution to 2D Euler equations.

Stream function:  psi(x, y) = log(cosh(y) + A * cos(x))
Velocity:         v1 = dpsi/dy,  v2 = -dpsi/dx
Vorticity:        omega = -(1 - A^2) / (cosh(y) + A * cos(x))^2

Parameters:
    A in (0, 1): controls vortex concentration.
        A -> 0: parallel shear flow (tanh profile)
        A -> 1: point vortices

Domain: periodic in x with period 2*pi, unbounded in y (but decays fast).
For numerics we truncate y to [-L_y, L_y] with L_y ~ 3-5.

This is a steady (time-independent) flow, so all quantities depend on (x, y) only.
The PINN will solve for omega_tilde(x, y, t) on top of this fixed base flow.
"""

import jax.numpy as jnp


def stream_function(x, y, A=0.5):
    """psi(x, y) = log(cosh(y) + A * cos(x))"""
    return jnp.log(jnp.cosh(y) + A * jnp.cos(x))


def velocity(x, y, A=0.5):
    """(v1, v2) from the stream function.

    v1 =  dpsi/dy = sinh(y) / (cosh(y) + A * cos(x))
    v2 = -dpsi/dx = A * sin(x) / (cosh(y) + A * cos(x))
    """
    denom = jnp.cosh(y) + A * jnp.cos(x)
    v1 = jnp.sinh(y) / denom
    v2 = A * jnp.sin(x) / denom
    return v1, v2


def vorticity(x, y, A=0.5):
    """omega(x, y) = -(1 - A^2) / (cosh(y) + A * cos(x))^2"""
    denom = jnp.cosh(y) + A * jnp.cos(x)
    return -(1.0 - A**2) / denom**2


def velocity_gradients(x, y, A=0.5):
    """Analytic spatial derivatives of v = (v1, v2).

    Returns: dv1_dx, dv1_dy, dv2_dx, dv2_dy
    """
    C = jnp.cos(x)
    S = jnp.sin(x)
    ch = jnp.cosh(y)
    sh = jnp.sinh(y)
    D = ch + A * C
    D2 = D**2

    # v1 = sinh(y) / D
    dv1_dx = A * S * sh / D2
    dv1_dy = (ch * D - sh * sh) / D2  # = (ch*(ch + A*C) - sh^2) / D^2

    # v2 = A * sin(x) / D
    dv2_dx = A * (C * D + A * S**2) / D2  # = A*(C*(ch+A*C) + A*S^2) / D^2
    dv2_dy = -A * S * sh / D2

    return dv1_dx, dv1_dy, dv2_dx, dv2_dy


def vorticity_gradients(x, y, A=0.5):
    """Analytic spatial derivatives of omega.

    omega = -(1 - A^2) / D^2,  D = cosh(y) + A*cos(x)

    d(omega)/dx = 2*(1-A^2)*A*sin(x) / D^3
    d(omega)/dy = -2*(1-A^2)*sinh(y) / D^3

    Returns: domega_dx, domega_dy
    """
    D = jnp.cosh(y) + A * jnp.cos(x)
    D3 = D**3
    coeff = 2.0 * (1.0 - A**2)

    domega_dx = coeff * A * jnp.sin(x) / D3
    domega_dy = -coeff * jnp.sinh(y) / D3
    return domega_dx, domega_dy
