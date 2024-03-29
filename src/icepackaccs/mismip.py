#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

"""
Define some constants for the MISMIP+ setup
"""
from firedrake import SpatialCoordinate, max_value, Constant, exp

Lx = 640.0e3
Ly = 80.0e3

a = Constant(0.3)


def mismip_bed_topography(mesh):
    if mesh.cell_dimension() == 2:
        x, y = SpatialCoordinate(mesh)
    else:
        x, y, z = SpatialCoordinate(mesh)

    x_c = Constant(300e3)
    X = x / x_c

    B_0 = Constant(-150)
    B_2 = Constant(-728.8)
    B_4 = Constant(343.91)
    B_6 = Constant(-50.57)
    B_x = B_0 + B_2 * X ** 2 + B_4 * X ** 4 + B_6 * X ** 6

    f_c = Constant(4e3)
    d_c = Constant(500)
    w_c = Constant(24e3)

    B_y = d_c * (1 / (1 + exp(-2 * (y - Ly / 2 - w_c) / f_c)) + 1 / (1 + exp(+2 * (y - Ly / 2 + w_c) / f_c)))

    z_deep = Constant(-720)
    return max_value(B_x + B_y, z_deep)
