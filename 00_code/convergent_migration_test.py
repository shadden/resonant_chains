import numpy as np
import sys
from matplotlib import pyplot as plt
sys.path.insert(0,"../00_code/")

import argparse
from celmech.disturbing_function import list_resonance_terms
from resonant_chains import get_chain_rebound_sim, get_chain_hpert, ResonantChainPoissonSeries
from resonant_chain_utils import relative_damping_rate_tau_wave
from scipy.optimize import linear_sum_assignment
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
    return p.parse_args()

def main():
    args = parse_args()
    system = args.system
    max_order = args.max_order
    save_path = data_path+f"{system}/{system}_dissipation_info.npz"
    eq_data = np.load(data_path+f"{system}/{system}_maxorder{max_order}.npz")
    data = np.load(data_path+f"{system}/elliptic_eq_data.npz")
    hpert = get_chain_hpert(eq_data['resonances'],eq_data['masses'],max_order,1)
    rc = ResonantChainPoissonSeries(
        eq_data['resonances'],
        eq_data['masses'],
        hpert,max_order = max_order
    )
    i_dK2 = 2 * rc.N_planar + rc.M
    Npl = rc.N_planar
    Ninf = np.ones(rc.N_planar) * np.inf
    i_min = np.argmin(np.abs(data['dK2vals']))
    dK2min = data['dK2vals'][i_min]
    eq_min = data['eqs'][i_min]
    rc.dK2 = dK2min
    X = np.insert(eq_min,i_dK2,dK2min)

    

    alpha_vals = np.linspace(0,3,20)
    dK2dot_all_migrate = np.zeros(alpha_vals.shape)
    dK2dot_inner_halted = np.zeros(alpha_vals.shape)
    for i,alpha in enumerate(alpha_vals):
        gamma_rel_a, gamma_rel_e = relative_damping_rate_tau_wave(eq_data['masses'],eq_data['resonances'],alpha = alpha)
        tau_a = 1/gamma_rel_a
        dK2dot_all_migrate[i] = rc.planar_flow_with_dissipation(X,Ninf,tau_a,0)[i_dK2]
        tau_a[0] = np.inf
        dK2dot_inner_halted[i] = rc.planar_flow_with_dissipation(X,Ninf,tau_a,0)[i_dK2]
    plt.plot(alpha_vals,dK2dot_all_migrate,label="All planets migrate")
    plt.plot(alpha_vals,dK2dot_inner_halted,label="Inner planets halted")
    plt.xlabel("alpha (surface density power law index)")
    plt.ylabel("dK2/dt")
    plt.legend()
    plt.show()

if __name__ =="__main__":
    main()