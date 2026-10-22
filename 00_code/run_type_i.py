#!/usr/bin/env python3
"""Three-planet Type-I migration into a REBOUNDx inner-disk-edge trap.

Example
-------
python run_type_i.py --tau-e 1000 --h 0.03 --ide-width 0.05 \
    --t-end 1000 --snapshots-per-tau-e 10 --output-prefix results/test

Units: Mstar=1, G=4*pi**2, disk-edge radius=1, orbital period at edge=1.
The tau-e argument is the linear damping time of planet 1 at a=1.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rebound
import reboundx

MASS_RATIOS = (1.38825302e-5, 1.03231440e-5, 1.16712670e-5)
PERIOD_RATIO = 1.5 * 1.01
EDGE_POSITION = 1.0


def surface_density_for_tau_e(tau_e, h, planet_mass, stellar_mass, G):
    """Sigma(1) from t_e=t_wave/0.78, evaluated at a=r=1."""
    return stellar_mass**1.5 * h**4 / (0.78 * planet_mass * np.sqrt(G) * tau_e)


def run(args):
    for name in ('tau_e', 'h', 'ide_width', 't_end', 'snapshots_per_tau_e'):
        if getattr(args, name) <= 0 or not np.isfinite(getattr(args, name)):
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive')
    if not np.isfinite(args.s) or not np.isfinite(args.beta):
        raise ValueError('Disk exponents must be finite')

    sim = rebound.Simulation()
    sim.G = 4 * np.pi**2
    sim.add(m=1.0)

    # Just outside the edge of the region where the migration torque is modified.
    a = EDGE_POSITION * (1.0 + 1.01 * args.ide_width)
    initial_axes = []
    for mass_ratio in MASS_RATIOS:
        sim.add(m=mass_ratio, a=a, e=0.0, inc=0.0)
        initial_axes.append(a)
        a *= PERIOD_RATIO ** (2.0 / 3.0)
    sim.move_to_com()

    sim.integrator = 'whfast'
    # At r=1 with G=4*pi^2 and Mstar=1, the Keplerian period is 1.
    P_edge = 2 * np.pi * np.sqrt(EDGE_POSITION**3 / (sim.G * sim.particles[0].m))
    sim.dt = P_edge / 20.0

    rebx = reboundx.Extras(sim)
    mig = rebx.load_force('type_I_migration')
    rebx.add_force(mig)

    sigma_1 = surface_density_for_tau_e(
        args.tau_e, args.h, sim.particles[1].m, sim.particles[0].m, sim.G
    )
    mig.params['tIm_surface_density_1'] = float(sigma_1)
    mig.params['tIm_scale_height_1'] = float(args.h)
    mig.params['tIm_surface_density_exponent'] = float(args.s)
    mig.params['tIm_flaring_index'] = float(args.beta)
    mig.params['ide_position'] = EDGE_POSITION
    mig.params['ide_width'] = float(args.ide_width)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    archive = prefix.with_suffix('.sa')
    rebx_file = prefix.with_suffix('.bin')
    metadata_file = prefix.with_suffix('.json')
    if not args.overwrite:
        existing = [p for p in (archive, rebx_file, metadata_file) if p.exists()]
        if existing:
            raise FileExistsError(f'Output already exists: {existing}. Use --overwrite to replace.')

    t_end = args.t_end * args.tau_e
    snapshot_interval = args.tau_e / args.snapshots_per_tau_e
    metadata = {
        'mass_ratios': list(MASS_RATIOS),
        'initial_semimajor_axes': initial_axes,
        'initial_period_ratios': [PERIOD_RATIO, PERIOD_RATIO],
        'initial_eccentricities': [0.0] * 3,
        'initial_inclinations': [0.0] * 3,
        'G': sim.G,
        'stellar_mass': sim.particles[0].m,
        'tau_e_reference': args.tau_e,
        'tau_e_definition': 'linear planet-1 damping time at a=r=1',
        'h_at_r1': args.h,
        'surface_density_exponent': args.s,
        'flaring_index': args.beta,
        'surface_density_at_r1': sigma_1,
        'ide_position': EDGE_POSITION,
        'ide_width_fraction': args.ide_width,
        'dt': sim.dt,
        't_end': t_end,
        'snapshots_per_tau_e': args.snapshots_per_tau_e,
        'snapshot_interval': snapshot_interval,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + '\n')
    rebx.save_to_file(str(rebx_file))

    # Save an initial snapshot, then register automatic periodic snapshots.
    sim.save_to_file(str(archive), delete_file=True)
    sim.save_to_file(str(archive), interval=snapshot_interval, delete_file=False)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f'Integrating to t={t_end:g}; archive={archive}', flush=True)
    sim.integrate(t_end, exact_finish_time=0)
    # Ensure the final state is archived, even if not on the snapshot schedule.
    sim.save_to_file(str(archive), delete_file=False)
    print(f'Finished at t={sim.t:g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tau-e', type=float, required=True, help='Linear planet-1 eccentricity damping time at r=1')
    parser.add_argument('--h', type=float, required=True, help='Aspect ratio H/r at r=1')
    parser.add_argument('--ide-width', type=float, required=True, help='Fractional inner disk edge width')
    parser.add_argument('--s', type=float, default=1.0, help='Surface density power-law index (default: 1)')
    parser.add_argument('--beta', type=float, default=0.25, help='Aspect-ratio flaring index (default: 0.25)')
    parser.add_argument('--t-end', type=float, default=1000.0, help='Final time in reference tau_e (default: 1000)')
    parser.add_argument('--snapshots-per-tau-e', type=float, default=10.0, help='Snapshot cadence (default: 10 per tau_e)')
    parser.add_argument('--output-prefix', type=str, required=True, help='Output prefix for .sa, .bin and .json')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    run(parser.parse_args())

if __name__ == '__main__':
    main()
