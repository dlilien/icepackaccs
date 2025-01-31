#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 dlilien <dlilien@noatak>
#
# Distributed under terms of the MIT license.

"""

"""
import icepack
import firedrake
from icepack.constants import year, ideal_gas as R
import numpy as np
from operator import itemgetter


A0_consts = {3: {"cold": 3.985e-13 * year * 1.0e18, "warm": 1.916e3 * year * 1.0e18},
             4: {"cold": 4.0e5 * year, "warm": 6.0e28 * year},
             1.8: {"cold": 3.9e-3 * year, "warm": 3.0e26 * year}}
Q_consts = {3: {"cold": 60, "warm": 139},
            4: {"cold": 60, "warm": 181},
            1.8: {"cold": 49, "warm": 192}}
trans_temps = {3: 263.15, 4: 258, 1.8: 255}


def rate_factor(T, n=3, m=None, m_exp=1.8):
    r"""Compute the rate factor in Glen's flow law for a given temperature

    The strain rate :math:`\dot\varepsilon` of ice resulting from a stress
    :math:`\tau` is

    .. math::
       \dot\varepsilon = A(T)\tau^3

    where :math:`A(T)` is the temperature-dependent rate factor:

    .. math::
       A(T) = A_0\exp(-Q/RT)

    where :math:`R` is the ideal gas constant, :math:`Q` has units of
    energy per mole, and :math:`A_0` is a prefactor with units of
    pressure :math:`\text{MPa}^{-3}\times\text{yr}^{-1}`.

    Parameters
    ----------
    T : float, np.ndarray, or UFL expression
        The ice temperature
    n : int, optional
        The flow law exponent
    m : float, np.ndarray, or UFL expression
        Grain size in meters
    m_exp : float
        Grain size exponents

    Returns
    -------
    A : same type as T
        The ice fluidity
    """
    import ufl

    if isinstance(T, ufl.core.expr.Expr):
        cold = firedrake.lt(T, trans_temps[n])
        A0 = firedrake.conditional(cold, A0_consts[n]["cold"], A0_consts[n]["warm"])
        Q = firedrake.conditional(cold, Q_consts[n]["cold"], Q_consts[n]["warm"])
        if m is not None:
            A = A0 * firedrake.exp(-Q / (R * T)) * m ** m_exp
        else:
            A = A0 * firedrake.exp(-Q / (R * T))
        if isinstance(T, firedrake.Constant):
            return firedrake.Constant(A)

        return A

    cold = T < trans_temps[n]
    warm = ~cold if isinstance(T, np.ndarray) else (not cold)
    A0 = A0_consts[n]["cold"] * cold + A0_consts[n]["warm"] * warm
    Q = Q_consts[n]["cold"] * cold + Q_consts[n]["warm"] * warm
    if m is not None:
        A0 * np.exp(-Q / (R * T)) * m ** m_exp
    else:
        return A0 * np.exp(-Q / (R * T))


def A_times_eps(**kwargs):
    r"""Return the viscous part of the hybrid model action functional

    The viscous component of the action for the hybrid model is

    .. math::
        E(u) = \frac{n}{n + 1}\int_\Omega\int_0^1\left(
        M : \dot\varepsilon_x + \tau_z\cdot\varepsilon_z\right)h\, d\zeta\; dx

    where :math:`M(\dot\varepsilon, A)` is the membrane stress tensor and
    :math:`\tau_z` is the vertical shear stress vector.

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can implement your own viscosity functional and pass it
    as an argument when initializing model objects to use your functional
    instead.

    Keyword arguments
    -----------------
    velocity : firedrake.Function
    surface : firedrake.Function
    thickness : firedrake.Function
    fluidity : firedrake.Function
        `A` in Glen's flow law

    Returns
    -------
    firedrake.Form
    """
    u, h, s, A = itemgetter("velocity", "thickness", "surface", "fluidity")(kwargs)
    ε_min = kwargs.get("strain_rate_min", firedrake.Constant(1.0e-10))
    n = kwargs.get("n", 3)

    ε_x = icepack.models.hybrid.horizontal_strain_rate(velocity=u, surface=s, thickness=h)
    ε_z = icepack.models.hybrid.vertical_strain_rate(velocity=u, thickness=h)
    ε_e = icepack.models.hybrid._effective_strain_rate(ε_x, ε_z, ε_min)
    return A * ε_e ** (n), ε_e
