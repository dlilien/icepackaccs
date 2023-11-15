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


def tau_regularized_coulomb_mismip(m, u, h, s, α2=0.5, β2=1.0e-2):
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


regularized_coulomb = get_regularized_coulomb()
regularized_coulomb_mismip = get_regularized_coulomb_mismip()
