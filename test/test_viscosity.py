import math

import firedrake
import numpy as np
import pytest

from icepackaccs import viscosity


def test_flow_law_conversion_helpers():
    assert np.isclose(
        viscosity.nondim_A(2.0, 3.0),
        2.0 / (viscosity.EPS_SCALE / viscosity.TAU_SCALE**3),
    )

    for n in [1.8, 3.0, 3.5, 4.0]:
        A_axial = 2.5
        A_octal = viscosity.axial_to_octahedral(A_axial, n)
        assert np.isclose(
            viscosity.axial_to_effective(A_axial, n),
            viscosity.octahedral_to_effective(A_octal, n),
        )
        assert A_octal > 0


def test_rate_factor_scalar_and_array_paths():
    scalar = viscosity.rate_factor(263.15, n=3)
    array = viscosity.rate_factor(np.array([250.0, 263.15, 270.0]), n=3)
    grain_size = viscosity.rate_factor(255.0, n=1.8, m=0.001)

    assert math.isfinite(float(scalar))
    assert scalar > 0
    assert np.all(np.isfinite(array))
    assert np.all(array > 0)
    assert math.isfinite(float(grain_size))
    assert grain_size > 0


def test_rate_factor_firedrake_paths(depth_averaged_fields, assert_finite_assemble):
    mesh = depth_averaged_fields["mesh"]
    scalar_space = depth_averaged_fields["fluidity"].function_space()

    constant = viscosity.rate_factor(firedrake.Constant(263.15), n=3)
    assert math.isfinite(float(constant))

    temperature = firedrake.Function(scalar_space).assign(263.15)
    rate = firedrake.Function(scalar_space).interpolate(
        viscosity.rate_factor(temperature, n=3)
    )
    assert_finite_assemble(rate, mesh)


def test_viscosity_is_extruded(depth_averaged_fields, hybrid_fields):
    assert viscosity._is_extruded(depth_averaged_fields["velocity"]) is False
    assert viscosity._is_extruded(hybrid_fields["velocity"]) is True


def test_effective_strain_rate_2d(depth_averaged_fields, assert_finite_assemble):
    mesh = depth_averaged_fields["mesh"]
    strain_rate = viscosity.effective_strain_rate(depth_averaged_fields["velocity"])
    assert_finite_assemble(strain_rate, mesh)


def test_effective_strain_rate_extruded(hybrid_fields, assert_finite_assemble):
    mesh = hybrid_fields["mesh"]
    strain_rate = viscosity.effective_strain_rate(
        velocity=hybrid_fields["velocity"],
        thickness=hybrid_fields["thickness"],
        surface=hybrid_fields["surface"],
    )
    assert_finite_assemble(strain_rate, mesh)


def test_effective_strain_rate_argument_errors(depth_averaged_fields, hybrid_fields):
    with pytest.raises(TypeError):
        viscosity.effective_strain_rate()

    with pytest.raises(TypeError):
        viscosity.effective_strain_rate(
            depth_averaged_fields["velocity"],
            velocity=depth_averaged_fields["velocity"],
        )

    with pytest.raises(ValueError):
        viscosity.effective_strain_rate(hybrid_fields["velocity"])


def test_A3_to_An_helpers(depth_averaged_fields, assert_finite_assemble):
    mesh = depth_averaged_fields["mesh"]
    scalar_space = depth_averaged_fields["fluidity"].function_space()
    kwargs = depth_averaged_fields

    A3 = firedrake.Function(scalar_space).assign(20.0)
    An = viscosity.A3_to_An(
        A3,
        kwargs["velocity"],
        kwargs["thickness"],
        kwargs["surface"],
        4.0,
        scalar_space,
    )
    nondim_An = viscosity.nondim_A3_to_An(
        A3,
        kwargs["velocity"],
        kwargs["thickness"],
        kwargs["surface"],
        4.0,
        scalar_space,
    )

    assert An.function_space() == scalar_space
    assert nondim_An.function_space() == scalar_space
    assert_finite_assemble(An, mesh)
    assert_finite_assemble(nondim_An, mesh)


def test_A_times_eps_hybrid(hybrid_fields, assert_finite_assemble):
    mesh = hybrid_fields["mesh"]
    expression, strain_rate = viscosity.A_times_eps(**hybrid_fields)

    assert_finite_assemble(expression, mesh)
    assert_finite_assemble(strain_rate, mesh)


@pytest.mark.parametrize(
    "functional",
    [
        viscosity.tunable_viscosity,
        viscosity.tunable_depth_averaged_viscosity,
        viscosity.depth_averaged_viscosity,
        viscosity.nondim_viscosity_depth_averaged,
        viscosity.nondim_tunable_depth_averaged_viscosity,
    ],
)
def test_depth_averaged_viscosity_functionals(
    functional, depth_averaged_fields, assert_finite_assemble
):
    mesh = depth_averaged_fields["mesh"]
    expression = functional(**depth_averaged_fields)
    assert_finite_assemble(expression, mesh)
