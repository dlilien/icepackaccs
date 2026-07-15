import json
import math

import firedrake

from icepackaccs import friction
from icepackaccs.reprojection import extract_bed, extract_surface, interpolate2d3d


def _assemble(expr, mesh):
    value = firedrake.assemble(
        expr * firedrake.dx(domain=mesh),
        form_compiler_parameters={"quadrature_degree": 8},
    )
    return float(value)


def _function_summary(function):
    mesh = function.ufl_domain()
    coordinates = firedrake.SpatialCoordinate(mesh)
    if function.ufl_shape == ():
        weight = 1.0 + sum((i + 1) * coordinates[i] for i in range(len(coordinates)))
        values = [
            _assemble(function, mesh),
            _assemble(function * function, mesh),
            _assemble(function * weight, mesh),
        ]
    else:
        weight = firedrake.as_vector(
            tuple(1.0 + (i + 1) * coordinates[i] for i in range(function.ufl_shape[0]))
        )
        values = [
            _assemble(firedrake.inner(function, function), mesh),
            _assemble(firedrake.inner(function, weight), mesh),
        ]

    assert all(math.isfinite(value) for value in values)
    return values


def _build_fields():
    base_mesh = firedrake.UnitSquareMesh(2, 2)
    extruded_mesh = firedrake.ExtrudedMesh(base_mesh, layers=1)
    x, y, z = firedrake.SpatialCoordinate(extruded_mesh)

    scalar_space_2d = firedrake.FunctionSpace(base_mesh, "CG", 2)
    scalar_space_extruded = firedrake.FunctionSpace(
        extruded_mesh, "CG", 2, vfamily="CG", vdegree=1
    )
    bed_scalar_space = firedrake.FunctionSpace(
        extruded_mesh, "CG", 2, vfamily="R", vdegree=0
    )
    vector_space_extruded = firedrake.VectorFunctionSpace(
        extruded_mesh, "CG", 2, dim=2, vfamily="CG", vdegree=1
    )
    bed_vector_space = firedrake.VectorFunctionSpace(
        extruded_mesh, "CG", 2, dim=2, vfamily="R", vdegree=0
    )

    scalar_2d = firedrake.Function(scalar_space_2d).interpolate(
        1.0
        + firedrake.SpatialCoordinate(base_mesh)[0]
        + 2.0 * firedrake.SpatialCoordinate(base_mesh)[1]
    )
    scalar_extruded = firedrake.Function(scalar_space_extruded).interpolate(
        1.0 + x + 2.0 * y + 3.0 * z
    )
    vector_extruded = firedrake.Function(vector_space_extruded).interpolate(
        firedrake.as_vector((1.0 + x + z, 2.0 + y + 2.0 * z))
    )
    bed_velocity = firedrake.Function(bed_vector_space).interpolate(
        firedrake.as_vector((100.0 + x, 50.0 + y))
    )
    bed_friction = firedrake.Function(bed_scalar_space).interpolate(
        10.0 + x + 2.0 * y
    )

    return {
        "extruded_mesh": extruded_mesh,
        "scalar_2d": scalar_2d,
        "scalar_extruded": scalar_extruded,
        "vector_extruded": vector_extruded,
        "bed_velocity": bed_velocity,
        "bed_friction": bed_friction,
    }


def main():
    fields = _build_fields()
    C3 = friction.c1_to_c3(fields["bed_friction"], fields["bed_velocity"])
    beta = friction.c3_to_beta(C3, fields["bed_velocity"], u0=300.0)
    C1 = friction.c3_to_c1(C3, fields["bed_velocity"], minslide=1.0)

    results = {
        "friction.c1_to_c3": _function_summary(C3),
        "friction.c3_to_beta": _function_summary(beta),
        "friction.c3_to_c1": _function_summary(C1),
        "reprojection.extract_bed.scalar": _function_summary(
            extract_bed(fields["scalar_extruded"])
        ),
        "reprojection.extract_surface.scalar": _function_summary(
            extract_surface(fields["scalar_extruded"])
        ),
        "reprojection.extract_bed.vector": _function_summary(
            extract_bed(fields["vector_extruded"])
        ),
        "reprojection.extract_surface.vector": _function_summary(
            extract_surface(fields["vector_extruded"])
        ),
        "reprojection.interpolate2d3d": _function_summary(
            interpolate2d3d(fields["scalar_2d"], fields["extruded_mesh"])
        ),
    }

    if firedrake.COMM_WORLD.rank == 0:
        print("MPI_DAT_ACCESS_RESULTS " + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
