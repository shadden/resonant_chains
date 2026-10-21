from pathlib import Path
import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(0,"../00_code/")
from scipy.interpolate import interp1d
import argparse
from celmech.disturbing_function import list_resonance_terms
from resonant_chains import get_chain_rebound_sim, get_chain_hpert, ResonantChainPoissonSeries
from resonant_chain_utils import relative_damping_rate_tau_wave
import sympy as sp
data_path = "/Users/hadden/Papers/10_chain_dynamics/03_data/"
system_name = "Kepler-60"
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute whether predicted migration rates are consistent with chain formation via convergent migration"
    )
    p.add_argument(
        "--system",
        default="Kepler-60",
        help="System to analyze"
    )
    p.add_argument(
        "--max_order",
        default=3,
        type=int,
        help="Maximum order for the resonant chain Hamiltonian expansion"
    )
    p.add_argument(
        "--h0_order",
        default=2,
        help="Maximum order of expansion for the Keplerian part of the Hamiltonian"
    )
    return p.parse_args()

def main():
    args = parse_args()
    args = parse_args()
    system_name = args.system
    max_order = args.max_order
    h0_order = args.h0_order
    cache_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") / system_name
    sys_data = np.load(cache_dir / f"{system_name}_maxorder{max_order}_h0order{h0_order}.npz")
    eq_data = np.load(cache_dir / f"elliptic_eq_data_maxorder{max_order}_h0order{h0_order}.npz")
    hpert = get_chain_hpert(sys_data['resonances'],sys_data['masses'],max_order)
    rc = ResonantChainPoissonSeries(
        sys_data['resonances'],
        sys_data['masses'],
        hpert,max_order = max_order, h0_order= h0_order
    )
    i_dK2 = 2 * rc.N_planar + rc.M
    Ninf = np.ones(rc.N_planar) * np.inf
    i_min = np.argmin(np.abs(eq_data['dK2vals']))
    dK2min = eq_data['dK2vals'][i_min]
    eq_min = eq_data['eqs'][i_min]
    rc.dK2 = dK2min
    X = np.insert(eq_min,i_dK2,dK2min)

    

    alpha_vals = np.linspace(0,3,20)
    dK2dot_all_migrate = np.zeros(alpha_vals.shape)
    dK2dot_inner_halted = np.zeros(alpha_vals.shape)
    for i,alpha in enumerate(alpha_vals):
        gamma_rel_a, gamma_rel_e = relative_damping_rate_tau_wave(sys_data['masses'],sys_data['resonances'],alpha = alpha)
        tau_a = 1/gamma_rel_a
        print(f"alpha={alpha:.2f}, tau_a = {tau_a}")
        dK2dot_all_migrate[i] = rc.planar_flow_with_dissipation(X,Ninf,tau_a,0)[i_dK2]
        tau_a[0] = np.inf
        dK2dot_inner_halted[i] = rc.planar_flow_with_dissipation(X,Ninf,tau_a,0)[i_dK2]
    if np.any(dK2dot_inner_halted<0):
        alpha_crit = interp1d(dK2dot_inner_halted,alpha_vals,fill_value='extrapolate')(0)
        print(f"Convergent migration w/ inner halted can produce the observed equilibrium at alpha = {alpha_crit:.2f}")
    if np.any(dK2dot_all_migrate<0):
        alpha_crit = interp1d(dK2dot_all_migrate,alpha_vals,fill_value='extrapolate')(0)
        print(f"Standard migration can produce the observed equilibrium at alpha = {alpha_crit:.2f}")

    plt.plot(alpha_vals,dK2dot_all_migrate,label="All planets migrate")
    plt.plot(alpha_vals,dK2dot_inner_halted,label="Inner planets halted")
    plt.xlabel("alpha (surface density power law index)")
    plt.ylabel("dK2/dt")
    plt.legend()
    plt.show()

if __name__ =="__main__":
    main()