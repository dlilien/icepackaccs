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

# 1.8 and 4 from Fan et al.'s recallibration of Goldsby & Kohlstedt; this avoids discontinuities in A(T)
# 3 follows standard icepack
# 3.5 follows Fan et al., 2025 for high stress

# 1.8, 3.5, 4 are octahedral stress/strain
# We need to convert each of these to be in terms of effective stress
# 3 is already in terms of effective stress



def axial_to_octahedral(A_axial, n):
    # S7 from Fan et al., 2025
    return A_axial * 3. ** n / 2. ** ((n + 1) / 2.0)


def octahedral_to_effective(A_oct, n):
    # S7 from Fan et al., 2025
    return A_oct * (2. / 3.) ** ((n - 1) / 2.0)


def axial_to_effective(A_axial, n):
    A_oct = axial_to_octahedral(A_axial, n)
    return octahedral_to_effective(A_oct, n)


A0_consts = {
    3: {"cold": 3.985e-13 * year * 1.0e18,
        "warm": 1.916e3 * year * 1.0e18},
    3.5: {"cold": octahedral_to_effective(10**12.89 * year, 3.5),
          "warm": octahedral_to_effective(10**12.89 * year, 3.5)},
    4: {"cold": octahedral_to_effective(10**6.85 * year, 4),
        "warm": octahedral_to_effective(10**25 * year, 4)},
    1.8: {"cold": octahedral_to_effective(10**2.48 * year, 1.8),
          "warm": octahedral_to_effective(10**38.37 * year, 1.8)},
    # 4: {"cold": axial_to_effective(4.0e5 * year, 4),  # Uncorrected Goldsby & Kohlstedt
    #     "warm": axial_to_effective(6.0e28 * year, 4)},
    # 1.8: {"cold": axial_to_effective(3.9e-3 * year, 1.8),
    #       "warm": axial_to_effective(3.0e26 * year, 1.8)},
}
Q_consts = {
    3: {"cold": 60, "warm": 139},
    3.5: {"cold": 90, "warm": 90},
    4: {"cold": 60, "warm": 181},
    1.8: {"cold": 49, "warm": 192},
}
trans_temps = {
    3: 263.15,
    3.5: 0,  # One value, we choose to call everything warm
    4: 262,
    1.8: 262
    # 4: 258,  # Uncorrected Goldsby & Kohlstedt
    # 1.8: 255,  # Uncorrected Goldsby & Kohlstedt
}


def rate_factor(T, n=3, m=None, m_exp=-1.4, trans_temps=trans_temps):
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
            A = A0 * firedrake.exp(-Q / (R * T)) * m**m_exp
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
        return A0 * np.exp(-Q / (R * T)) * m**m_exp
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    T = np.linspace(-30, 0, 100) + 273.15
    # ax.plot(T, rate_factor(T, n=3), label="$n$=3")
    ax.plot(T, rate_factor(T, n=4, trans_temps={4: 400}), label="$n$=4 cold")
    ax.plot(T, rate_factor(T, n=4, trans_temps={4: 10}), label="$n$=4 warm")
    # ax.plot(T, rate_factor(T, n=1.8, m=0.001), label="$n$=1.8")
    ax.legend(loc="best")
    plt.xlabel("Temperature (K)")
    plt.ylabel("A(T) (MPa$^{-4}$ yr$^{-1}$)")
    fig.savefig("/Users/dlilien/Desktop/A_of_T_n=4.pdf")

    fig, ax = plt.subplots()
    T0 = -10 + 273.15
    T2 = -12 + 273.15
    tau = 10 ** np.linspace(-3, 0, 100)

    A3 = rate_factor(T0, n=3)
    A3_5 = rate_factor(T0, n=3.5)
    A4 = rate_factor(T0, n=4)
    A1_8 = rate_factor(T0, n=1.8, m=0.001)
    ax.plot(tau, tau**3 * A3, label="$n$=3")
    ax.plot(tau, tau**3.5 * A3_5, label="$n$=3.5")
    ax.plot(tau, tau**4 * A4, label="$n$=4")
    ax.plot(tau, tau**1.8 * A1_8, label="$n$=1.8")

    A3 = rate_factor(T2, n=3)
    A3_5 = rate_factor(T2, n=3.5)
    A4 = rate_factor(T2, n=4)
    A1_8 = rate_factor(T2, n=1.8, m=0.001)
    ax.plot(tau, tau**3 * A3, label="$n$=3", linestyle="dashed")
    ax.plot(tau, tau**3.5 * A3_5, label="$n$=3.5", linestyle="dashed")
    ax.plot(tau, tau**4 * A4, label="$n$=4", linestyle="dashed")
    ax.plot(tau, tau**1.8 * A1_8, label="$n$=1.8", linestyle="dashed")

    ax.legend(loc="best")

    plt.show()
