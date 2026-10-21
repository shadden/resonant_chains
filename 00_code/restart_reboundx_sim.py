
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

    # Read the eccentricity damping timescale.
    tau_e = abs(inner.params["tau_e"])

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
