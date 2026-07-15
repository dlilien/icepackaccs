import math

import firedrake
import pytest


@pytest.fixture(scope="session")
def mesh2d():
    return firedrake.UnitSquareMesh(2, 2)


@pytest.fixture(scope="session")
def extruded_mesh(mesh2d):
    return firedrake.ExtrudedMesh(mesh2d, layers=1)


@pytest.fixture
def depth_averaged_fields(mesh2d):
    x, y = firedrake.SpatialCoordinate(mesh2d)
    vector_space = firedrake.VectorFunctionSpace(mesh2d, "CG", 2)
    scalar_space = firedrake.FunctionSpace(mesh2d, "CG", 2)

    velocity = firedrake.Function(vector_space).interpolate(
        firedrake.as_vector((100.0 + x, 50.0 + y))
    )
    thickness = firedrake.Function(scalar_space).assign(100.0)
    surface = firedrake.Function(scalar_space).assign(120.0)
    fluidity = firedrake.Function(scalar_space).assign(20.0)
    friction = firedrake.Function(scalar_space).assign(10.0)
    mod_A = firedrake.Function(scalar_space).assign(0.01)
    is_floating = firedrake.Function(scalar_space).assign(1.0)

    return {
        "mesh": mesh2d,
        "velocity": velocity,
        "thickness": thickness,
        "surface": surface,
        "fluidity": fluidity,
        "friction": friction,
        "mod_A": mod_A,
        "is_floating": is_floating,
    }


@pytest.fixture
def hybrid_fields(extruded_mesh):
    x, y, z = firedrake.SpatialCoordinate(extruded_mesh)
    vector_space = firedrake.VectorFunctionSpace(
        extruded_mesh, "CG", 1, dim=2, vfamily="CG", vdegree=1
    )
    scalar_space = firedrake.FunctionSpace(
        extruded_mesh, "CG", 1, vfamily="CG", vdegree=1
    )

    velocity = firedrake.Function(vector_space).interpolate(
        firedrake.as_vector((100.0 + x + z, 50.0 + y + z))
    )
    thickness = firedrake.Function(scalar_space).assign(100.0)
    surface = firedrake.Function(scalar_space).assign(120.0)
    fluidity = firedrake.Function(scalar_space).assign(20.0)

    return {
        "mesh": extruded_mesh,
        "velocity": velocity,
        "thickness": thickness,
        "surface": surface,
        "fluidity": fluidity,
    }


@pytest.fixture
def extruded_bed_fields(extruded_mesh):
    x, y, _ = firedrake.SpatialCoordinate(extruded_mesh)
    vector_space = firedrake.VectorFunctionSpace(
        extruded_mesh, "CG", 2, dim=2, vfamily="R", vdegree=0
    )
    scalar_space = firedrake.FunctionSpace(
        extruded_mesh, "CG", 2, vfamily="R", vdegree=0
    )

    velocity = firedrake.Function(vector_space).interpolate(
        firedrake.as_vector((100.0 + x, 50.0 + y))
    )
    friction = firedrake.Function(scalar_space).assign(10.0)

    return {
        "mesh": extruded_mesh,
        "velocity": velocity,
        "friction": friction,
    }


@pytest.fixture
def assert_finite_assemble():
    def _assert_finite_assemble(expr, mesh):
        value = firedrake.assemble(
            expr * firedrake.dx(domain=mesh),
            form_compiler_parameters={"quadrature_degree": 6},
        )
        assert math.isfinite(float(value))
        return float(value)

    return _assert_finite_assemble
