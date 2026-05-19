import celmech as cm
from celmech.disturbing_function import list_resonance_terms
from celmech.poisson_series import DFTerm_as_PSterms
import rebound as rb
from celmech.poisson_series import PoissonSeries
from celmech.poisson_series import PSTerm
from scipy.special import comb
import numpy as np

def get_3br_sim(masses,resonance_1,resonance_2):
    sim = rb.Simulation()
    sim.add(m=1)
    sim.add(m=masses[0],a=1)
    j1, k1 = resonance_1 
    sim.add(m=masses[1],P = (j1/(j1-k1) * sim.particles[1].P))
    j2, k2 = resonance_2
    sim.add(m=masses[2],P = (j2/(j2-k2) * sim.particles[2].P))
    sim.move_to_com()
    return sim

def generalized_binom(n, k):
    if k < 0:
        return 0
    if isinstance(n, int) and n < 0:
        return (-1)**k * comb(-n + k - 1, k, exact=True)
    return comb(n, k)  # falls back to normal definition for non-negative n

def h0_series(pham,max_order = 2):
    L0s = np.array([pham.H_params[L0] for L0 in pham.Lambda0s[1:]])
    n0s = np.array([p.n for p in pham.particles[1:]])
    terms = []
    Npl = pham.N-1
    zero_N = np.zeros(2*Npl,dtype=int)
    zero_M = np.zeros(Npl,dtype=int)
    for l in range(1,max_order+1):
        for n0,L0,o_i in zip(n0s,L0s,np.eye(Npl)):
            term = PSTerm(
                -0.5 * n0 * L0**(1-l)*generalized_binom(-2,l),
                zero_N,
                zero_N,
                l * o_i,
                zero_M
            )
            terms.append(term)
    return PoissonSeries.from_PSTerms(terms)

def get_omega_series(pham):
    h0 = h0_series(pham)
    Npl = pham.N-1
    zero_N = np.zeros(2*Npl,dtype=int)
    zero_M = np.zeros(Npl,dtype=int)
    omega_series = [PSTerm(-1j,zero_N,zero_N,zero_M,-1*o_i).as_series()*h0.Lie_deriv(PSTerm(1,zero_N,zero_N,zero_M,o_i).as_series()) for  o_i in np.eye(Npl,dtype=int)]
    return omega_series

def get_3br_sim(masses,Delta1,resonance_1,resonance_2):
    sim = rb.Simulation()
    sim.add(m=1)
    sim.add(m=masses[0],a=1)
    P1 = sim.particles[1].P
    j1, k1 = resonance_1 
    P2 = (j1/(j1-k1) * P1) * (1+Delta1)
    sim.add(m=masses[1],P = P2 )
    j2, k2 = resonance_2
    kvec = np.array([0,k2-j2,j2]) - np.array([k1-j1,j1,0])
    P3  = - 1 * kvec[2] / (kvec[0] / P1 + kvec[1] / P2)
    sim.add(m=masses[2],P = P3)
    sim.move_to_com()
    return sim

eval_at_zero = lambda ps: ps(np.zeros(ps.N),np.zeros(ps.M),np.zeros(ps.M))
def get_hres_series(pham,i,j,res_jk,**kwargs):
    h_res = []
    for k,nu in list_resonance_terms(*res_jk,**kwargs):
        for lvec in ((0,0),(1,0),(0,1)):
            h_res+=DFTerm_as_PSterms(pham,i,j,k,nu,lvec)
    h_res=cm.poisson_series.PoissonSeries.from_PSTerms(h_res)
    return h_res

def hpert_to_chi_terms(hpert,omega_series):
    chi_terms = []
    for term in hpert.terms:
        chi_terms.append(
            (-1j * term.as_series(), 
             np.sum([k_i * omega_i for k_i,omega_i in zip(term.q,omega_series)])
            )
        )
    return chi_terms
from resonant_chains import resonant_chain_variables_transformation_matrix
def get_n0_Q3br_dndL_Minv(masses,Delta_in,resonance_in,resonance_out):
    j_in,k_in = resonance_in
    j_out,k_out = resonance_out
    kvec = np.array([j_in - k_in, k_out - j_out - j_in, j_out])
    sim = get_3br_sim(masses,Delta_in,resonance_in,resonance_out)
    pham = cm.PoincareHamiltonian(cm.Poincare.from_Simulation(sim))
    omega_series = get_omega_series(pham)
    n0 = np.real([eval_at_zero(omega_i) for omega_i in omega_series])    
    h_inner = get_hres_series(pham,1,2,resonance_in)
    h_outer = get_hres_series(pham,2,3,resonance_out)
    h1 = h_inner + h_outer
    chi_terms = hpert_to_chi_terms(h1,omega_series)
    bracket_series = PoissonSeries(h1.N,h1.M)
    for num, denom in chi_terms:
        N_denom = eval_at_zero(denom)
        term = -0.5 * ( h1.Lie_deriv(num) * (1/N_denom) + -1 * num * (1/N_denom)**2 * h1.Lie_deriv(num) )
        bracket_series += term
    Q3br = PoissonSeries.from_PSTerms([term for term in bracket_series.terms if np.all(term.p==0) and np.all(term.q==kvec)])
    Q3br *=2
    dn_dL = np.real([w_series.terms[1].C  for w_series in omega_series]) 
    Minv = -1 * kvec**2 @ dn_dL
    return n0,Q3br,dn_dL,Minv
