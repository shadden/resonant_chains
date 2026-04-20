import numpy as np
import sys
sys.path.insert(0,"../00_code/")

import argparse
from celmech.disturbing_function import list_resonance_terms
from resonant_chains import get_chain_rebound_sim, get_chain_hpert, ResonantChainPoissonSeries
from scipy.optimize import linear_sum_assignment
import sympy as sp
data_path = "/Users/hadden/Papers/10_chain_dynamics/03_data/"
system_name = "Kepler-60"
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute damping rate coefficient for resonant chain equilibria"
    )
    p.add_argument(
        "--system",
        default="Kepler-60",
        help="System to analyze"
    )
    return p.parse_args()

def main():
    args = parse_args()
    system = args.system
    save_path = data_path+f"{system}/{system}_dissipation_info.npz"
    eq_data = np.load(data_path+f"{system}/{system}_maxorder3.npz")
    data = np.load(data_path+f"{system}/elliptic_eq_data.npz")
    hpert = get_chain_hpert(eq_data['resonances'],eq_data['masses'],3,1)
    rc = ResonantChainPoissonSeries(
        eq_data['resonances'],
        eq_data['masses'],
        hpert,max_order = 3
    )
    i_kappa2 = rc.N_planar 
    Npl = rc.N_planar
    Ninf = np.ones(rc.N_planar) * np.inf
    damping_rate_coeffs = np.zeros((data['eqs'].shape[0],(rc.N_planar + rc.M) + 1,Npl))

    for i,(dK2,z_eq,freqs) in enumerate(zip(data['dK2vals'],data['eqs'],data['freqs'])):
        
        z_extended_eq = np.insert(z_eq,[-rc.M],[dK2])
        # note-- planar jacobian w/ dissipation takes dK2 as argument.
        # no need to update dK2 value of rc instance
        j_cons_extended = rc.planar_jacobian_with_dissipation(
            z_extended_eq,
            Ninf, # tau_e
            Ninf, # tau_a
            1 # p
        )

        # get eigenvectors and values
        eigs_extended,vecsT_extended = np.linalg.eig(j_cons_extended)
        vecs_extended = vecsT_extended.T

        # match extended phase space frequencies
        extended_col_id = np.array([np.argmin(np.abs(np.imag(eigs_extended)-f)) for f in freqs])
        zero_indx = np.argmin(np.abs(eigs_extended))
        extended_col_id = np.append(extended_col_id,zero_indx)     
        
        # ensure unique column ids
        assert len(extended_col_id) == len(np.unique(extended_col_id)), "failed to identify unique column ids"

        # extended phase space vectors and covectors
        covecs_extended = np.linalg.inv(vecs_extended).T
        vecs_extended = vecs_extended[extended_col_id]
        covecs_extended = covecs_extended[extended_col_id]
        
        # test covector/vector pairing 
        assert np.all(np.isclose(covecs_extended @ vecs_extended.T,np.eye(covecs_extended.shape[0]))), "vector/covector mismatch!"

        j_dis_symb = rc.planar_dissipation_jacobian_symbolic(
            np.insert(z_extended_eq,i_kappa2,0)
        )
        j_dis_symb = np.delete(np.delete(j_dis_symb,i_kappa2,axis=0),i_kappa2,axis=1)
        damping_rates = [sp.re(_) for _ in np.diag(covecs_extended @ j_dis_symb @ vecs_extended.T)]
        damping_rate_coeffs[i] = np.array([
            [rate.coeff(g) for g in sp.symbols("gamma_e(1:{})".format(Npl+1),real=True)]
            for rate in damping_rates
        ],dtype=float)    
    np.savez(save_path,damping_rate_coeffs=damping_rate_coeffs)

if __name__ =="__main__":
    main()