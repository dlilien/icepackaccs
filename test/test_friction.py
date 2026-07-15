import math

import firedrake
import numpy as np
import pytest

from icepackaccs import friction


def test_friction_is_extruded(depth_averaged_fields, extruded_bed_fields):
    assert friction._is_extruded(depth_averaged_fields["velocity"]) is False
    assert friction._is_extruded(extruded_bed_fields["velocity"]) is True


def test_numpy_tau_helpers():
    u = np.array([10.0, 100.0, 300.0])
    h = np.array([100.0, 100.0, 100.0])
    s = np.array([120.0, 120.0, 120.0])

    regularized = friction.tau_regularized_coulomb_mismip(3.0, u, h, s)
    assay_davis = friction.tau_mismip_assaydavis(3.0, u, h, s)

    assert regularized.shape == u.shape
    assert assay_davis.shape == u.shape
    assert np.all(np.isfinite(regularized))
    assert np.all(np.isfinite(assay_davis))


def test_firedrake_tau_regularized_coulomb_mismip(
    depth_averaged_fields, assert_finite_assemble
):
    mesh = depth_averaged_fields["mesh"]
    tau = friction.tau_regularized_coulomb_mismip(
        3.0,
        depth_averaged_fields["velocity"],
        depth_averaged_fields["thickness"],
        depth_averaged_fields["surface"],
        fd=True,
    )
    assert_finite_assemble(firedrake.inner(tau, tau), mesh)


def test_friction_stress(depth_averaged_fields, assert_finite_assemble):
    mesh = depth_averaged_fields["mesh"]
    tau = friction.friction_stress(
        depth_averaged_fields["velocity"],
        depth_averaged_fields["friction"],
        m=3.0,
    )
    assert_finite_assemble(firedrake.inner(tau, tau), mesh)


def test_bed_friction(depth_averaged_fields, assert_finite_assemble):
    mesh = depth_averaged_fields["mesh"]
    expression = friction.bed_friction(**depth_averaged_fields)
    assert_finite_assemble(expression, mesh)


@pytest.mark.parametrize(
    "functional",
    [
        friction.get_regularized_coulomb(),
        friction.get_regularized_coulomb_ramp(),
        friction.get_regularized_coulomb_simp(),
        friction.get_regularized_coulomb_mismip(),
        friction.get_smooth_weertman(),
        friction.get_ramp_weertman(),
        friction.get_weertman(),
        friction.regularized_coulomb,
        friction.regularized_coulomb_mismip,
        friction.smooth_weertman_m3,
        friction.smooth_weertman_linear,
        friction.weertman_m3,
        friction.weertman_linear,
    ],
)
def test_friction_law_functionals(
    functional, depth_averaged_fields, assert_finite_assemble
):
    mesh = depth_averaged_fields["mesh"]
    expression = functional(**depth_averaged_fields)
    assert_finite_assemble(expression, mesh)


def test_friction_conversion_helpers_2d(
    depth_averaged_fields, assert_finite_assemble
):
    mesh = depth_averaged_fields["mesh"]
    velocity = depth_averaged_fields["velocity"]
    C1 = depth_averaged_fields["friction"]

    C3 = friction.c1_to_c3(C1, velocity)
    beta = friction.c3_to_beta(C3, velocity, u0=300.0)
    C1_roundtrip = friction.c3_to_c1(C3, velocity, minslide=1.0)

    assert C3.ufl_domain() == mesh
    assert beta.ufl_domain() == mesh
    assert C1_roundtrip.ufl_domain() == mesh
    assert_finite_assemble(C3, mesh)
    assert_finite_assemble(beta, mesh)
    assert_finite_assemble(C1_roundtrip, mesh)


def test_friction_conversion_helpers_extruded(
    extruded_bed_fields, assert_finite_assemble
):
    mesh = extruded_bed_fields["mesh"]
    velocity = extruded_bed_fields["velocity"]
    C1 = extruded_bed_fields["friction"]

    C3 = friction.c1_to_c3(C1, velocity)
    beta = friction.c3_to_beta(C3, velocity, u0=300.0)
    C1_roundtrip = friction.c3_to_c1(C3, velocity, minslide=1.0)

    assert C3.ufl_domain() == mesh
    assert beta.ufl_domain() == mesh
    assert C1_roundtrip.ufl_domain() == mesh
    assert_finite_assemble(C3, mesh)
    assert_finite_assemble(beta, mesh)
    assert_finite_assemble(C1_roundtrip, mesh)
