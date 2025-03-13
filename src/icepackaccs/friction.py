#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

"""

"""
import numpy as np
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from firedrake import max_value, min_value, sqrt, inner
from icepack.models.friction import itemgetter


def friction_stress(u, C, m=3):
    r"""Compute the shear stress for a given sliding velocity"""
    return -C * sqrt(inner(u, u)) ** (1 / m - 1) * u


def bed_friction(m=3.0, **kwargs):
    r"""Return the bed friction part of the ice stream action functional

    Mimics icepack, but uses a variable m.

    The frictional part of the ice stream action functional is

    .. math::
       E(u) = -\frac{m}{m + 1}\int_\Omega\tau(u, C)\cdot u\; dx

    where :math:`\\tau(u, C)` is the basal shear stress

    .. math::
       \tau(u, C) = -C|u|^{1/m - 1}u
    """
    u, C = itemgetter("velocity", "friction")(kwargs)
    τ = friction_stress(u, C, m)
    return -m / (m + 1) * inner(τ, u)


def get_regularized_coulomb(m=3.0, u_0=300.0, h_t=50.0):
    def regularized_coulomb(**kwargs):
        u = kwargs["velocity"]
        H = kwargs["thickness"]
        h = kwargs["surface"]
        C = kwargs["friction"]

        h_af = max_value(h - H * (1 - ρ_I / ρ_W), 0)
        ramp = min_value(1, h_af / h_t)
        U = sqrt(inner(u, u))

        return C * ramp * ((u_0 ** (1 / m + 1) + U ** (1 / m + 1)) ** (m / (m + 1)) - u_0)

    return regularized_coulomb


def get_regularized_coulomb_ramp(m=3.0, u_0=300.0, h_t=50.0):
    def regularized_coulomb(**kwargs):
        u = kwargs["velocity"]
        C = kwargs["friction"]
        h = kwargs["thickness"]
        s = kwargs["surface"]
        U = sqrt(inner(u, u))

        p_W = ρ_W * g * max_value(0, h - s)
        p_I = ρ_I * g * h
        ϕ = 1 - p_W / p_I

        return C * ϕ * ((u_0 ** (1 / m + 1) + U ** (1 / m + 1)) ** (m / (m + 1)) - u_0)

    return regularized_coulomb


def get_regularized_coulomb_simp(m=3.0, u_0=300.0, h_t=50.0):
    def regularized_coulomb(**kwargs):
        u = kwargs["velocity"]
        C = kwargs["friction"]

        U = sqrt(inner(u, u))
        return C * ((u_0 ** (1 / m + 1) + U ** (1 / m + 1)) ** (m / (m + 1)) - u_0)

    return regularized_coulomb


def get_regularized_coulomb_mismip(m=3.0, α2=0.2):
    def regularized_coulomb_mismip(**kwargs):
        variables = ("velocity", "thickness", "surface", "friction")
        u, h, s, β2 = map(kwargs.get, variables)

        p_W = ρ_W * g * max_value(0, -(s - h))
        p_I = ρ_I * g * h
        N = max_value(0, p_I - p_W)
        τ_c = α2 * N
        u_c = (τ_c / β2) ** m
        u_b = sqrt(inner(u, u))

        return τ_c * ((u_c ** (1 / m + 1) + u_b ** (1 / m + 1)) ** (m / (m + 1)) - u_c)

    return regularized_coulomb_mismip


def tau_regularized_coulomb_mismip(m, u, h, s, α2=0.5, β2=1.0e-2, fd=False):
    if fd:
        p_W = ρ_W * g * max_value(0, -(s - h))
        p_I = ρ_I * g * h
        N = max_value(0, p_I - p_W)
        u_b = sqrt(inner(u, u))
    else:
        p_W = ρ_W * g * np.maximum(-(s - h), 0)
        p_I = ρ_I * g * h
        N = np.maximum(p_I - p_W, 0)
        u_b = np.abs(u)
    return α2 * N * u_b ** (1 / m - 1) / ((α2 * N / β2) ** (m + 1) + u_b ** (1 / m + 1)) ** (1 / (m + 1)) * u


def tau_mismip_assaydavis(m, u, h, s, α2=0.5, β2=1.0e-2):
    p_W = ρ_W * g * np.maximum(-(s - h), 0)
    p_I = ρ_I * g * h
    N = np.maximum(p_I - p_W, 0)
    u_b = np.abs(u)
    return α2 * N * u_b ** (1 / m - 1) / ((α2 * N / β2) ** m + u_b) ** (1 / m) * u


def get_smooth_weertman(m=3.0):
    def smooth_weertman(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        s = kwargs["surface"]
        C = kwargs["friction"]

        p_W = ρ_W * g * max_value(0, h - s)
        p_I = ρ_I * g * h
        ϕ = 1 - p_W / p_I

        return bed_friction(
            m=m,
            velocity=u,
            friction=C * ϕ,
        )
    return smooth_weertman


def get_ramp_weertman(m=3.0, h_t=50.0):
    def smooth_weertman(**kwargs):
        u = kwargs["velocity"]
        H = kwargs["thickness"]
        h = kwargs["surface"]
        C = kwargs["friction"]

        h_af = max_value(h - H * (1 - ρ_I / ρ_W), 0)
        ramp = min_value(1, h_af / h_t)
        return bed_friction(
            m=m,
            velocity=u,
            friction=C * ramp,
        )
    return smooth_weertman


def get_weertman(m=3.0):
    def weertman(**kwargs):
        u = kwargs["velocity"]
        C = kwargs["friction"]

        return bed_friction(
            m=m,
            velocity=u,
            friction=C,
        )
    return weertman


regularized_coulomb = get_regularized_coulomb()
regularized_coulomb_mismip = get_regularized_coulomb_mismip()
smooth_weertman_m3 = get_smooth_weertman()
smooth_weertman_linear = get_smooth_weertman(m=1.0)
weertman_m3 = get_weertman(m=3.0)
weertman_linear = get_weertman(m=1.0)
