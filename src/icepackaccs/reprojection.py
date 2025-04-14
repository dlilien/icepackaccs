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
from firedrake.__future__ import interpolate


def extract_surface(q_in):
    mesh_x = q_in.ufl_domain()._base_mesh
    shape = q_in.ufl_shape
    if len(shape) == 0:
        VLin = firedrake.FunctionSpace(q_in.ufl_domain(), "CG", 2, vfamily="CG", vdegree=1)
        q_targ = firedrake.assemble(interpolate(q_in, VLin))
        element_xz = q_targ.ufl_element()
        element_x = element_xz.sub_elements[0]
    else:
        VLin = firedrake.VectorFunctionSpace(q_in.ufl_domain(), "CG", 2, dim=shape[0], vfamily="CG", vdegree=1)
        q_targ = firedrake.assemble(interpolate(q_in, VLin))
        element_xz = q_targ.ufl_element()
        element_xy = element_xz.sub_elements[0].sub_elements[0]
        element_x = firedrake.VectorElement(element_xy, dim=shape[0])

    Q_x = firedrake.FunctionSpace(mesh_x, element_x)
    q_x = firedrake.Function(Q_x)
    if len(shape) > 0:
        q_x.dat.data[:] = q_targ.dat.data[1::2, :]
    else:
        q_x.dat.data[:] = q_targ.dat.data[1::2]
    return q_x


def extract_bed(q_in):
    mesh_x = q_in.ufl_domain()._base_mesh
    shape = q_in.ufl_shape
    if len(shape) == 0:
        VLin = firedrake.FunctionSpace(q_in.ufl_domain(), "CG", 2, vfamily="CG", vdegree=1)
        q_targ = firedrake.assemble(interpolate(q_in, VLin))
        element_xz = q_targ.ufl_element()
        element_x = element_xz.sub_elements[0]
    else:
        VLin = firedrake.VectorFunctionSpace(q_in.ufl_domain(), "CG", 2, dim=shape[0], vfamily="CG", vdegree=1)
        q_targ = firedrake.assemble(interpolate(q_in, VLin))
        element_xz = q_targ.ufl_element()
        element_xy = element_xz.sub_elements[0].sub_elements[0]
        element_x = firedrake.VectorElement(element_xy, dim=shape[0])

    Q_x = firedrake.FunctionSpace(mesh_x, element_x)
    q_x = firedrake.Function(Q_x)
    if len(shape) > 0:
        q_x.dat.data[:] = q_targ.dat.data[::2, :]
    else:
        q_x.dat.data[:] = q_targ.dat.data[::2]
    return q_x


def interpolate2d3d(u, mesh):
    """Go up in dimension, using the same types of elements.

    In the vertical, vfamily will be R and vdegree 0.

    Parameters
    ----------
    u: firedrake.Function
        The function to interpolate onto the new mesh.
    mesh: firedrake.Mesh
        The 3d mesh on which to interpolate.
    """
    element = u.ufl_element()
    V = firedrake.FunctionSpace(mesh, family=element.family(), degree=element.degree(), vfamily="R", vdegree=0)
    u3 = firedrake.Function(V)
    u3.dat.data[:] = u.dat.data_ro[:]
    return u3
