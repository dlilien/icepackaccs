#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

"""

"""
import firedrake
from operator import itemgetter
from icepack.calculus import Identity
from icepack.models.viscosity import _effective_strain_rate
from icepack.constants import glen_flow_law as n, strain_rate_min
from icepack.calculus import trace, sym_grad
from numpy import pi


def threed_stress(**kwargs):
    r"""Calculate the membrane stress for a given strain rate and
    fluidity"""
    u, A = itemgetter("velocity", "fluidity")(kwargs)
    ε2d = sym_grad(u)
    ε_min = firedrake.Constant(kwargs.get("strain_rate_min", strain_rate_min))
    ε_e = _effective_strain_rate(ε2d, ε_min)
    ε = firedrake.as_matrix(([ε2d[0, 0], ε2d[0, 1], 0], [ε2d[1, 0], ε2d[1, 1], 0], [0, 0, -ε2d[0, 0] - ε2d[1, 1]]))

    Ident = Identity(3)
    μ = 0.5 * A ** (-1 / n) * ε_e ** (1 / n - 1)
    return 2 * μ * (ε + trace(ε) * Ident)


def invariants_principal(A):
    """Principal invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3
    """
    i1 = firedrake.tr(A)
    i2 = (firedrake.tr(A) ** 2 - firedrake.tr(A * A)) / 2
    i3 = firedrake.det(A)
    return i1, i2, i3


def eigenstate3(A):
    eps = 3.0e-16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754

    I1, I2, I3 = invariants_principal(A)
    dq = 2 * I1 ** 3 - 9 * I1 * I2 + 27 * I3
    #
    Δx = [
        A[0, 1] * A[1, 2] * A[2, 0] - A[0, 2] * A[1, 0] * A[2, 1],
        A[0, 1] ** 2 * A[1, 2] - A[0, 1] * A[0, 2] * A[1, 1] + A[0, 1] * A[0, 2] * A[2, 2] - A[0, 2] ** 2 * A[2, 1],
        A[0, 0] * A[0, 1] * A[2, 1] - A[0, 1] ** 2 * A[2, 0] - A[0, 1] * A[2, 1] * A[2, 2] + A[0, 2] * A[2, 1] ** 2,
        A[0, 0] * A[0, 2] * A[1, 2] + A[0, 1] * A[1, 2] ** 2 - A[0, 2] ** 2 * A[1, 0] - A[0, 2] * A[1, 1] * A[1, 2],
        A[0, 0] * A[0, 1] * A[1, 2]
        - A[0, 1] * A[0, 2] * A[1, 0]
        - A[0, 1] * A[1, 2] * A[2, 2]
        + A[0, 2] * A[1, 2] * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 2] * A[2, 1]
        - A[0, 1] * A[0, 2] * A[2, 0]
        + A[0, 1] * A[1, 2] * A[2, 1]
        - A[0, 2] * A[1, 1] * A[2, 1],  # noqa: E501
        A[0, 1] * A[1, 0] * A[1, 2]
        - A[0, 2] * A[1, 0] * A[1, 1]
        + A[0, 2] * A[1, 0] * A[2, 2]
        - A[0, 2] * A[1, 2] * A[2, 0],  # noqa: E501
        A[0, 0] ** 2 * A[1, 2]
        - A[0, 0] * A[0, 2] * A[1, 0]
        - A[0, 0] * A[1, 1] * A[1, 2]
        - A[0, 0] * A[1, 2] * A[2, 2]
        + A[0, 1] * A[1, 0] * A[1, 2]
        + A[0, 2] * A[1, 0] * A[2, 2]
        + A[1, 1] * A[1, 2] * A[2, 2]
        - A[1, 2] ** 2 * A[2, 1],  # noqa: E501
        A[0, 0] ** 2 * A[1, 2]
        - A[0, 0] * A[0, 2] * A[1, 0]
        - A[0, 0] * A[1, 1] * A[1, 2]
        - A[0, 0] * A[1, 2] * A[2, 2]
        + A[0, 2] * A[1, 0] * A[1, 1]
        + A[0, 2] * A[1, 2] * A[2, 0]
        + A[1, 1] * A[1, 2] * A[2, 2]
        - A[1, 2] ** 2 * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 1]
        - A[0, 0] * A[0, 1] * A[2, 2]
        - A[0, 1] ** 2 * A[1, 0]
        + A[0, 1] * A[0, 2] * A[2, 0]
        - A[0, 1] * A[1, 1] * A[2, 2]
        + A[0, 1] * A[2, 2] ** 2
        + A[0, 2] * A[1, 1] * A[2, 1]
        - A[0, 2] * A[2, 1] * A[2, 2],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 1]
        - A[0, 0] * A[0, 1] * A[2, 2]
        + A[0, 0] * A[0, 2] * A[2, 1]
        - A[0, 1] ** 2 * A[1, 0]
        - A[0, 1] * A[1, 1] * A[2, 2]
        + A[0, 1] * A[1, 2] * A[2, 1]
        + A[0, 1] * A[2, 2] ** 2
        - A[0, 2] * A[2, 1] * A[2, 2],  # noqa: E501
        A[0, 0] * A[0, 1] * A[1, 2]
        - A[0, 0] * A[0, 2] * A[1, 1]
        + A[0, 0] * A[0, 2] * A[2, 2]
        - A[0, 1] * A[1, 1] * A[1, 2]
        - A[0, 2] ** 2 * A[2, 0]
        + A[0, 2] * A[1, 1] ** 2
        - A[0, 2] * A[1, 1] * A[2, 2]
        + A[0, 2] * A[1, 2] * A[2, 1],  # noqa: E501
        A[0, 0] * A[0, 2] * A[1, 1]
        - A[0, 0] * A[0, 2] * A[2, 2]
        - A[0, 1] * A[0, 2] * A[1, 0]
        + A[0, 1] * A[1, 1] * A[1, 2]
        - A[0, 1] * A[1, 2] * A[2, 2]
        + A[0, 2] ** 2 * A[2, 0]
        - A[0, 2] * A[1, 1] ** 2
        + A[0, 2] * A[1, 1] * A[2, 2],  # noqa: E501
        A[0, 0] ** 2 * A[1, 1]
        - A[0, 0] ** 2 * A[2, 2]
        - A[0, 0] * A[0, 1] * A[1, 0]
        + A[0, 0] * A[0, 2] * A[2, 0]
        - A[0, 0] * A[1, 1] ** 2
        + A[0, 0] * A[2, 2] ** 2
        + A[0, 1] * A[1, 0] * A[1, 1]
        - A[0, 2] * A[2, 0] * A[2, 2]
        + A[1, 1] ** 2 * A[2, 2]
        - A[1, 1] * A[1, 2] * A[2, 1]
        - A[1, 1] * A[2, 2] ** 2
        + A[1, 2] * A[2, 1] * A[2, 2],
    ]  # noqa: E501
    Δy = [
        A[0, 2] * A[1, 0] * A[2, 1] - A[0, 1] * A[1, 2] * A[2, 0],
        A[1, 0] ** 2 * A[2, 1] - A[1, 0] * A[1, 1] * A[2, 0] + A[1, 0] * A[2, 0] * A[2, 2] - A[1, 2] * A[2, 0] ** 2,
        A[0, 0] * A[1, 0] * A[1, 2] - A[0, 2] * A[1, 0] ** 2 - A[1, 0] * A[1, 2] * A[2, 2] + A[1, 2] ** 2 * A[2, 0],
        A[0, 0] * A[2, 0] * A[2, 1] - A[0, 1] * A[2, 0] ** 2 + A[1, 0] * A[2, 1] ** 2 - A[1, 1] * A[2, 0] * A[2, 1],
        A[0, 0] * A[1, 0] * A[2, 1]
        - A[0, 1] * A[1, 0] * A[2, 0]
        - A[1, 0] * A[2, 1] * A[2, 2]
        + A[1, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0] * A[1, 2] * A[2, 0]
        - A[0, 2] * A[1, 0] * A[2, 0]
        + A[1, 0] * A[1, 2] * A[2, 1]
        - A[1, 1] * A[1, 2] * A[2, 0],  # noqa: E501
        A[0, 1] * A[1, 0] * A[2, 1]
        - A[0, 1] * A[1, 1] * A[2, 0]
        + A[0, 1] * A[2, 0] * A[2, 2]
        - A[0, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0] ** 2 * A[2, 1]
        - A[0, 0] * A[0, 1] * A[2, 0]
        - A[0, 0] * A[1, 1] * A[2, 1]
        - A[0, 0] * A[2, 1] * A[2, 2]
        + A[0, 1] * A[1, 0] * A[2, 1]
        + A[0, 1] * A[2, 0] * A[2, 2]
        + A[1, 1] * A[2, 1] * A[2, 2]
        - A[1, 2] * A[2, 1] ** 2,  # noqa: E501
        A[0, 0] ** 2 * A[2, 1]
        - A[0, 0] * A[0, 1] * A[2, 0]
        - A[0, 0] * A[1, 1] * A[2, 1]
        - A[0, 0] * A[2, 1] * A[2, 2]
        + A[0, 1] * A[1, 1] * A[2, 0]
        + A[0, 2] * A[2, 0] * A[2, 1]
        + A[1, 1] * A[2, 1] * A[2, 2]
        - A[1, 2] * A[2, 1] ** 2,  # noqa: E501
        A[0, 0] * A[1, 0] * A[1, 1]
        - A[0, 0] * A[1, 0] * A[2, 2]
        - A[0, 1] * A[1, 0] ** 2
        + A[0, 2] * A[1, 0] * A[2, 0]
        - A[1, 0] * A[1, 1] * A[2, 2]
        + A[1, 0] * A[2, 2] ** 2
        + A[1, 1] * A[1, 2] * A[2, 0]
        - A[1, 2] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0] * A[1, 0] * A[1, 1]
        - A[0, 0] * A[1, 0] * A[2, 2]
        + A[0, 0] * A[1, 2] * A[2, 0]
        - A[0, 1] * A[1, 0] ** 2
        - A[1, 0] * A[1, 1] * A[2, 2]
        + A[1, 0] * A[1, 2] * A[2, 1]
        + A[1, 0] * A[2, 2] ** 2
        - A[1, 2] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0] * A[1, 0] * A[2, 1]
        - A[0, 0] * A[1, 1] * A[2, 0]
        + A[0, 0] * A[2, 0] * A[2, 2]
        - A[0, 2] * A[2, 0] ** 2
        - A[1, 0] * A[1, 1] * A[2, 1]
        + A[1, 1] ** 2 * A[2, 0]
        - A[1, 1] * A[2, 0] * A[2, 2]
        + A[1, 2] * A[2, 0] * A[2, 1],  # noqa: E501
        A[0, 0] * A[1, 1] * A[2, 0]
        - A[0, 0] * A[2, 0] * A[2, 2]
        - A[0, 1] * A[1, 0] * A[2, 0]
        + A[0, 2] * A[2, 0] ** 2
        + A[1, 0] * A[1, 1] * A[2, 1]
        - A[1, 0] * A[2, 1] * A[2, 2]
        - A[1, 1] ** 2 * A[2, 0]
        + A[1, 1] * A[2, 0] * A[2, 2],  # noqa: E501
        A[0, 0] ** 2 * A[1, 1]
        - A[0, 0] ** 2 * A[2, 2]
        - A[0, 0] * A[0, 1] * A[1, 0]
        + A[0, 0] * A[0, 2] * A[2, 0]
        - A[0, 0] * A[1, 1] ** 2
        + A[0, 0] * A[2, 2] ** 2
        + A[0, 1] * A[1, 0] * A[1, 1]
        - A[0, 2] * A[2, 0] * A[2, 2]
        + A[1, 1] ** 2 * A[2, 2]
        - A[1, 1] * A[1, 2] * A[2, 1]
        - A[1, 1] * A[2, 2] ** 2
        + A[1, 2] * A[2, 1] * A[2, 2],
    ]  # noqa: E501
    Δd = [9, 6, 6, 6, 8, 8, 8, 2, 2, 2, 2, 2, 2, 1]
    Δ = 0
    for i in range(len(Δd)):
        Δ += Δx[i] * Δd[i] * Δy[i]

    Δxp = [A[1, 0], A[2, 0], A[2, 1], -A[0, 0] + A[1, 1], -A[0, 0] + A[2, 2], -A[1, 1] + A[2, 2]]
    Δyp = [A[0, 1], A[0, 2], A[1, 2], -A[0, 0] + A[1, 1], -A[0, 0] + A[2, 2], -A[1, 1] + A[2, 2]]
    Δdp = [6, 6, 6, 1, 1, 1]

    dp = 0
    for i in range(len(Δdp)):
        dp += 1 / 2 * Δxp[i] * Δdp[i] * Δyp[i]

    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    dp += eps ** 2
    Δ += eps ** 2

    phi3 = firedrake.atan2(firedrake.sqrt(27.0) * firedrake.sqrt(Δ), dq)

    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ = [(I1 + 2 * firedrake.sqrt(dp) * firedrake.cos((phi3 + 2 * pi * k) / 3)) / 3 for k in range(1, 4)]
    #
    # --- determine eigenprojectors E0, E1, E2
    #
    # E = [firedrake.derivative(λk, A).T for λk in λ]

    return λ  # , E


def principal_stress(**kwargs):
    τ = threed_stress(**kwargs)
    return eigenstate3(τ)


def von_mises_stress(eigs):
    return firedrake.sqrt(((eigs[0] - eigs[1]) ** 2.0 + (eigs[0] - eigs[2]) ** 2.0 + (eigs[1] - eigs[2]) ** 2.0) / 2.0)
