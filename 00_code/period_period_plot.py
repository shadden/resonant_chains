#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

def find_systems_near_synodic_resonances(
    systems,
    highlight_dict=None,
    resonances=((2,1),(1, 1), (2, 3), (1, 2)),
    tol=0.02,
    max_hits_per_system=None,
):
    """
    Identify systems with points lying close to specified synodic resonance lines.

    Parameters
    ----------
    systems : iterable of (hostname, group_df)
        Typically the grouped object used by compute_highlight_points, e.g.
        filtered.groupby("hostname", sort=False)
    highlight_dict : dict or None
        Existing highlighted systems to exclude.
    resonances : iterable of (p, q)
        Synodic resonances p:q with line y = 1 + (p/q)*(1-x).
    tol : float
        Maximum absolute vertical distance |y - y_res(x)| for a point to count
        as "near" a resonance.
    max_hits_per_system : int or None
        If not None, cap the number of returned hits per system.

    Returns
    -------
    candidates : dict
        Mapping hostname -> list of hit dictionaries
    """
    if highlight_dict is None:
        highlight_dict = {}

    excluded = set(highlight_dict.keys())
    candidates = {}

    for hostname, group in systems:
        if hostname in excluded:
            continue

        g = group.sort_values("pl_orbper").reset_index(drop=True)
        periods = g["pl_orbper"].to_numpy()

        if len(periods) < 3:
            continue

        hits = []
        for i in range(1, len(periods) - 1):
            x = periods[i] / periods[i - 1]
            y = periods[i] / periods[i + 1]

            best = None
            for p, q in resonances:
                y_res = 1 + (p / q) * (1 - x)
                delta = y - y_res
                adelta = abs(delta)

                if adelta <= tol:
                    rec = {
                        "hostname": hostname,
                        "triple_index": i,
                        "x": x,
                        "y": y,
                        "resonance": f"{p}:{q}",
                        "delta": delta,
                        "abs_delta": adelta,
                    }
                    if best is None or adelta < best["abs_delta"]:
                        best = rec

            if best is not None:
                hits.append(best)

        if hits:
            hits.sort(key=lambda d: d["abs_delta"])
            if max_hits_per_system is not None:
                hits = hits[:max_hits_per_system]
            candidates[hostname] = hits

    return candidates

def fetch_young_systems(cache_file: str, refresh: bool) -> pd.DataFrame:
    """
    Query pscomppars for a fixed list of 'young' multi-planet systems
    (by hostname). Save to a CSV cache for reproducibility.
    """
    hostnames = [
        "V1298 Tau", "AU Mic", "TOI-713", "TOI-1136", "K2-136",
        "Kepler-289", "HD 63433", "Kepler-51", "Kepler-411",
        "TOI-2076", "TOI-451"
    ]
    hostlist = ",".join(f"'{h}'" for h in hostnames)

    if (not refresh) and os.path.exists(cache_file):
        print(f"[info] Loading cached young systems from {cache_file}")
        return pd.read_csv(cache_file)

    print("[info] Querying NASA Exoplanet Archive for young systems...")
    tab = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,hostname,pl_orbper,pl_rade,tran_flag",
        where=f"hostname in ({hostlist})",
        cache=False,
    )
    df = tab.to_pandas()

    df.to_csv(cache_file, index=False)
    print(f"[info] Saved young systems to {cache_file} ({len(df)} rows).")
    return df

def fetch_exoplanet_data(cache_file: str, refresh: bool) -> pd.DataFrame:
    """
    Load cached CSV if present and refresh==False; otherwise query the
    NASA Exoplanet Archive and write the CSV cache.
    We intentionally query *all* transiting planets and filter by radius
    per-system locally to ensure the 'all planets < 6 Re' constraint is correct.
    """
    if (not refresh) and os.path.exists(cache_file):
        print(f"[info] Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)

    print("[info] Querying NASA Exoplanet Archive (pscomppars, transiting only)...")
    tab = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,hostname,pl_orbper,pl_rade,tran_flag",
        where="tran_flag = 1",
        cache=False,
    )
    df = tab.to_pandas()

    # Save raw query for transparency/reproducibility
    df.to_csv(cache_file, index=False)
    print(f"[info] Saved fresh query to {cache_file} ({len(df)} rows).")
    return df


def filter_compact_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only systems (grouped by hostname) that:
      - have at least 3 transiting planets,
      - have no missing orbital periods or radii,
      - and ALL planets in the system have pl_rade < 6 Earth radii.
    """
    def _valid_group(g: pd.DataFrame) -> bool:
        if len(g) < 3:
            return False
        if g["pl_orbper"].isnull().any() or g["pl_rade"].isnull().any():
            return False
        return (g["pl_rade"] < 6).all()

    grouped = df.groupby("hostname", sort=False)
    filtered = grouped.filter(_valid_group)
    return filtered


def compute_period_ratio_points(filtered: pd.DataFrame):
    """
    For each system (sorted by period), for each internal planet i (1..N-2):
      x = P_i / P_{i-1}
      y = P_i / P_{i+1}
    Returns x_vals, y_vals (np.ndarray), and the grouped object for reuse.
    """
    systems = filtered.groupby("hostname", sort=False)
    x_vals, y_vals = [], []
    for _, group in systems:
        g = group.sort_values("pl_orbper")
        P = g["pl_orbper"].to_numpy()
        if len(P) < 3:
            continue
        for i in range(1, len(P) - 1):
            x_vals.append(P[i] / P[i - 1])
            y_vals.append(P[i] / P[i + 1])
    return np.array(x_vals), np.array(y_vals), systems

def compute_highlight_points(systems, highlight_ordered_list):
    """
    Return an OrderedDict: system_name -> [(x1, y1), (x2, y2), ...],
    preserving the input order.
    """
    highlight_data = OrderedDict()
    system_map = {name: group for name, group in systems}

    for name in highlight_ordered_list:
        if name not in system_map:
            continue
        group = system_map[name].sort_values("pl_orbper")
        P = group["pl_orbper"].to_numpy()
        if len(P) < 3:
            continue
        xy = [(P[i] / P[i - 1], P[i] / P[i + 1]) for i in range(1, len(P) - 1)]
        highlight_data[name] = xy

    return highlight_data

def add_synodic_resonance_guides(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xvals = np.linspace(*xlim, 500)

    resonances = [
        (2,1),
        (1, 1),
        (2, 3),
        (1, 2),
        (1, 3)
    ]
    ylabs = [
        0.55,
        0.55,
        0.55,
        0.55,
        0.8
    ]
    for (p, q),ylab in zip(resonances,ylabs):
        yvals = 1 + (p / q) * (1 - xvals)

        # Plot the resonance line
        ax.plot(xvals, yvals, color='red', lw=2, zorder=1)

        # Keep only the visible part for labeling
        mask = (yvals >= ylim[0]) & (yvals <= ylim[1])
        if not np.any(mask):
            continue

        idx = np.where(mask)[0]
        i0, i1 = idx[0], idx[-1]
        mid_idx = idx[len(idx) // 2]

        xlab = 1 + q/p - q*ylab/p

        # Compute label angle in display coordinates
        p0 = ax.transData.transform((xvals[i0], yvals[i0]))
        p1 = ax.transData.transform((xvals[i1], yvals[i1]))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

        ax.text(
            xlab, ylab,
            f"{p}:{q}",
            color='red',
            fontsize=9,
            ha='center',
            va='center',
            rotation=angle,
            rotation_mode='anchor',
            bbox=dict(
                facecolor='white',
                edgecolor='red',
                boxstyle='round,pad=0.2'
            ),
            zorder=100
        )

def add_resonance_guides(ax, first_order_j=(2, 3, 4, 5,6), second_order_j=(5, 7)):
    """
    Draw resonance guide lines.

    First-order:
        x = j/(j-1), y = (j-1)/j   labeled j:j-1

    Second-order:
        x = j/(j-2), y = (j-2)/j   labeled j:j-2
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # First-order MMRs: 2:1, 3:2, 4:3, 5:4
    for j in first_order_j:
        xres = j / (j - 1)
        yres = (j - 1) / j

        ax.axvline(xres, color='k', lw=1)
        ax.axhline(yres, color='k', lw=1)

        if xlim[0] <= xres <= xlim[1]:
            ax.text(
                xres + 0.01, ylim[0] + 0.01, f"{j}:{j-1}",
                va='bottom', ha='left', rotation=90, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
            )
        if ylim[0] <= yres <= ylim[1]:
            ax.text(
                xlim[0] + 0.01, yres + 0.005, f"{j}:{j-1}",
                va='bottom', ha='left', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
            )

    # Second-order MMRs: 3:1, 5:3, 7:5, ...
    for j in second_order_j:
        xres = j / (j - 2)
        yres = (j - 2) / j

        ax.axvline(xres, color='gray', lw=1, ls='--')
        ax.axhline(yres, color='gray', lw=1, ls='--')

        if xlim[0] <= xres <= xlim[1]:
            ax.text(
                xres + 0.01, ylim[0] + 0.04, f"{j}:{j-2}",
                va='bottom', ha='left', rotation=90, fontsize=9, color='gray',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2')
            )
        if ylim[0] <= yres <= ylim[1]:
            ax.text(
                xlim[0] + 0.01, yres + 0.005, f"{j}:{j-2}",
                va='bottom', ha='left', fontsize=9, color='gray',
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2')
            )

import itertools
import matplotlib.colors as mcolors

import itertools
import matplotlib.colors as mcolors
GOLDEN = 0.5 * (1 + np.sqrt(5))
import matplotlib.gridspec as gridspec

def compute_all_period_ratios(df: pd.DataFrame) -> np.ndarray:
    """
    Compute all adjacent period ratios (P_i / P_{i-1}) for every
    system with at least two planets.
    """
    ratios = []
    systems = df.groupby("hostname", sort=False)
    for _, group in systems:
        g = group.sort_values("pl_orbper")
        P = g["pl_orbper"].dropna().to_numpy()
        if len(P) < 2:
            continue
        ratios.extend(P[1:] / P[:-1])
    return np.array(ratios)

def plot_period_ratio_plane(x_vals, y_vals,
                            highlight_dict=None,
                            xlim=(1, 2.2), ylim=(1/2.2, 1),
                            save_path=None):
    """
    Plot P_i/P_{i-1} vs P_i/P_{i+1} for compact multi-planet systems.
    """
    fig, ax = plt.subplots(figsize=(6 * GOLDEN, 6))

    # Main scatter plot
    ax.scatter(x_vals, y_vals, color='0.2', s=5, zorder=20)

    handles = []
    labels = []

    if highlight_dict:
        color_cycle = itertools.cycle(plt.get_cmap("tab10").colors)

        # Slightly vary sizes so coincident points are easier to distinguish
        size_cycle = itertools.cycle(np.linspace(70,30,len(highlight_dict)))

        # Sort so the most-multi-point systems get drawn first
        # and the later ones remain visible on top
        items = sorted(highlight_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

        for i, (system, xy_list) in enumerate(items):
            xs, ys = zip(*xy_list)
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)

            color = next(color_cycle)
            size = next(size_cycle)
            #dx, dy = next(offset_cycle)

            # Apply tiny display-space-inspired data offset
            # so exact overlaps become distinguishable
            xs_plot = xs #+ dx
            ys_plot = ys #+ dy

            sc = ax.scatter(
                xs_plot, ys_plot,
                s=size,
                color=color,
                edgecolor='none',
                linewidth=0.7,
                alpha=1,
                marker='o',
                label=system,
                zorder=100 + i
            )
            handles.append(sc)
            labels.append(system)

    ax.set_xlabel(r'$P_i / P_{i-1}$')
    ax.set_ylabel(r'$P_i / P_{i+1}$')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    add_resonance_guides(ax)
    add_synodic_resonance_guides(ax)

    ax.set_title("Period Ratios of Multi-transiting Systems")

    if highlight_dict:
        plt.subplots_adjust(right=0.75)
        leg = ax.legend(
            handles, labels,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=9, title="Resonant Chains"
        )
        leg.set_zorder(200)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[info] Saved figure to {save_path}")
    else:
        plt.show()

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot P_i/P_{i-1} vs P_i/P_{i+1} for compact multi-planet transiting systems (all R < 6 R_earth)."
    )
    p.add_argument(
        "--cache-file",
        default="exoplanet_transiting_pscomppars.csv",
        help="Path to CSV cache for Exoplanet Archive query (default: %(default)s)"
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force a fresh query to the Exoplanet Archive, overwriting the cache."
    )
    p.add_argument(
        "--highlight",
        default=" HD 110067, K2-138, Kepler-60, Kepler-80, Kepler-223, TOI-178, TOI-1136, TRAPPIST-1",
        help="Comma-separated hostnames to highlight (default: %(default)s)."
    )
    p.add_argument(
        "--save-plot",
        default="",
        help="If provided, save the plot to this path instead of showing it (e.g., plot.png or plot.pdf)."
    )
    p.add_argument(
        "--xlim",
        default="1.19,2.1",
        help="X-axis limits as 'xmin,xmax' (default: 1.19,2.2)."
    )
    p.add_argument(
        "--ylim",
        default=f"{1/2.1:.5f},{1/1.2:.5f}",
        help=f"Y-axis limits as 'ymin,ymax' (default: {1/2.2:.5f},{1/1.2:.5f})."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Parse limits
    xlim = tuple(float(v) for v in args.xlim.split(","))
    ylim = tuple(float(v) for v in args.ylim.split(","))

    # Fetch + cache data, then filter
    df = fetch_exoplanet_data(args.cache_file, args.refresh)
    filtered = filter_compact_multiples(df)

    if filtered.empty:
        print("[warn] No systems met the filter criteria (>=3 planets, all R<6 Re, no missing periods/radii).")
        return

    # Compute points
    x_vals, y_vals, systems = compute_period_ratio_points(filtered)

    # Highlights
    highlight_hosts = [h.strip() for h in args.highlight.split(",") if h.strip()]
    highlight_dict = compute_highlight_points(systems, highlight_hosts)

    # Plot
    plot_period_ratio_plane(
        x_vals, y_vals,
        highlight_dict=highlight_dict,
        xlim=xlim,
        ylim=ylim,
        save_path=args.save_plot if args.save_plot else None
    )
    if False:
        candidate_systems = find_systems_near_synodic_resonances(
        systems,
            highlight_dict=highlight_dict,
            resonances=((1, 1), (2, 3), (1, 2)),
            tol=0.005,
            max_hits_per_system=1,
        )
        for host, hits in sorted(candidate_systems.items(), key=lambda kv: kv[1][0]["abs_delta"]):
            h = hits[0]
            print(f"{host:20s}  {h['resonance']:>3s}  "
                f"x={h['x']:.3f}  y={h['y']:.3f}  |Δ|={h['abs_delta']:.4f}")
    
if __name__ == "__main__":
    main()
