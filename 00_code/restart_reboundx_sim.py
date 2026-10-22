
#!/usr/bin/env python3
"""
Restart and extend a REBOUND simulation with REBOUNDx forces.

Usage
-----
python extend_simulation.py simulation.sa reboundx.bin 1000

The final argument specifies the absolute integration end time
in units of the innermost planet's eccentricity damping timescale.
"""

import argparse
from pathlib import Path

import rebound
import reboundx


import numpy as np


def type_I_tau_e(sim, inner, force, linear=True):
    """
    Calculate the eccentricity damping timescale from REBOUNDx's
    type_I_migration disk parameters.

    Parameters
    ----------
    sim : rebound.Simulation
    inner : rebound.Particle
        Planet for which to calculate tau_e.
    force : reboundx.Force
        Loaded type_I_migration force.
    linear : bool
        If True, return the small-eccentricity, zero-inclination
        damping timescale.

    Returns
    -------
    float
        Eccentricity damping timescale in simulation time units.
    """
    star = sim.particles[0]
    orbit = inner.orbit(primary=star)
    a, e, inc = orbit.a, orbit.e, orbit.inc

    s = force.params["tIm_surface_density_exponent"]
    beta = force.params["tIm_flaring_index"]
    sigma_1 = force.params["tIm_surface_density_1"]
    h_1 = force.params["tIm_scale_height_1"]

    # Evaluate the disk profiles at the planet's semimajor axis.
    sigma = sigma_1 * a**(-s)
    h = h_1 * a**beta

    t_wave = (
        star.m**1.5 * h**4
        / (inner.m * sigma * np.sqrt(sim.G * a))
    )

    tau_e = t_wave / 0.78

    if not linear:
        eh = e / h
        ih = inc / h
        tau_e *= (
            1.0 - 0.14 * eh**2
            + 0.06 * eh**3
            + 0.18 * eh * ih**2
        )

    return tau_e


def get_inner_tau_e(sim, extras, inner):
    """
    Determine the inner planet's eccentricity damping timescale
    from either tau_e particle parameters or type_I_migration.
    """
    # First try explicitly specified eccentricity damping.
    try:
        return abs(inner.params["tau_e"])
    except (KeyError, AttributeError):
        pass

    # Otherwise look for the Type I migration force.
    try:
        force = extras.get_force("type_I_migration")
    except (KeyError, AttributeError, RuntimeError):
        raise ValueError(
            "Could not determine the eccentricity damping timescale: "
            "the inner planet has no tau_e parameter and "
            "type_I_migration could not be retrieved."
        )
    return type_I_tau_e(sim, inner, force)

def extend_simulation(
    archive_file,
    reboundx_file,
    t_end_tau_e,
    snapshots_per_tau_e=10.0,
):
    archive_file = Path(archive_file)
    reboundx_file = Path(reboundx_file)

    # Load the last snapshot of the Simulationarchive.
    sa = reboundx.Simulationarchive(str(archive_file),str(reboundx_file))
    sim,extras = sa[-1]

    # Get the innermost planet (assuming particle 0 is the star).
    inner = min(
        sim.particles[1:sim.N_active or sim.N],
        key=lambda p: p.a
    )

    # Determine tau_e from the appropriate damping prescription.
    tau_e = get_inner_tau_e(sim, extras, inner)

    # Convert the requested final time to simulation units.
    t_end = t_end_tau_e * tau_e

    # Set the snapshot interval based on the requested number of snapshots per tau_e.
    if snapshots_per_tau_e <= 0:
        raise ValueError("Snapshots per tau_e must be positive.")
    snapshot_interval = tau_e / snapshots_per_tau_e

    if t_end <= sim.t:
        raise ValueError(
            f"Requested final time ({t_end:.6g}) must exceed "
            f"the current simulation time ({sim.t:.6g})."
        )

    print(f"Archive:        {archive_file}")
    print(f"REBOUNDx file:  {reboundx_file}")
    print(f"Current time:   {sim.t / tau_e:.6g} tau_e")
    print(f"Final time:     {t_end_tau_e:.6g} tau_e")
    print(f"tau_e:          {tau_e:.6g}")
    print(f"Snapshots/tau_e: {snapshots_per_tau_e:.6g}")

    # Continue writing snapshots to the existing archive.
    # The original snapshot interval is retained.
    sim.save_to_file(
        str(archive_file),
        interval=snapshot_interval,
        delete_file=False
        )

    # Extend the integration.
    sim.integrate(t_end)

    print("Integration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extend a REBOUND/REBOUNDx Simulationarchive."
    )

    parser.add_argument(
        "archive",
        help="REBOUND Simulationarchive file"
    )
    parser.add_argument(
        "reboundx",
        help="REBOUNDx binary save file"
    )
    parser.add_argument(
        "t_end",
        type=float,
        help="Absolute final time in units of inner planet tau_e"
    )
    parser.add_argument(
        "--snapshots-per-tau-e",
        type=float,
        default=10.0,
        metavar="N",
        help="Number of snapshots per inner planet tau_e (default: 10)",
    )
    args = parser.parse_args()

    extend_simulation(args.archive, args.reboundx, args.t_end, args.snapshots_per_tau_e)
