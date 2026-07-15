import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_MPI_ENVIRONMENT_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_RANK",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_SIZE",
)
_RESULT_PREFIX = "MPI_DAT_ACCESS_RESULTS "
_FRICTION_KEYS = (
    "friction.c1_to_c3",
    "friction.c3_to_beta",
    "friction.c3_to_c1",
)
_REPROJECTION_KEYS = (
    "reprojection.extract_bed.scalar",
    "reprojection.extract_surface.scalar",
    "reprojection.extract_bed.vector",
    "reprojection.extract_surface.vector",
    "reprojection.interpolate2d3d",
)


def _inside_mpi_launcher():
    return any(key in os.environ for key in _MPI_ENVIRONMENT_KEYS)


def _run_probe(rank_count):
    repo_root = Path(__file__).resolve().parents[1]
    probe = Path(__file__).with_name("mpi_dat_access_probe.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    )
    env.pop("PYTEST_CURRENT_TEST", None)

    completed = subprocess.run(
        ["mpiexec", "-n", str(rank_count), sys.executable, str(probe)],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    for line in completed.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line.removeprefix(_RESULT_PREFIX))

    pytest.fail(
        "MPI dat-access probe did not emit results.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.fixture(scope="session")
def dat_access_rank_results():
    return _run_probe(1), _run_probe(2)


@pytest.mark.skipif(
    _inside_mpi_launcher(),
    reason="this driver test launches its own one-rank and two-rank MPI jobs",
)
def test_friction_dat_access_results_match_between_one_and_two_ranks(
    dat_access_rank_results,
):
    one_rank, two_ranks = dat_access_rank_results

    for key in _FRICTION_KEYS:
        assert key in one_rank
        assert key in two_ranks
        assert two_ranks[key] == pytest.approx(
            one_rank[key], rel=1.0e-12, abs=1.0e-12
        )


@pytest.mark.skipif(
    _inside_mpi_launcher(),
    reason="this driver test launches its own one-rank and two-rank MPI jobs",
)
def test_reprojection_dat_access_results_match_between_one_and_two_ranks(
    dat_access_rank_results,
):
    one_rank, two_ranks = dat_access_rank_results

    for key in _REPROJECTION_KEYS:
        assert key in one_rank
        assert key in two_ranks
        assert two_ranks[key] == pytest.approx(
            one_rank[key], rel=1.0e-12, abs=1.0e-12
        )
