import numpy as np
from resonant_chains import *
import rebound as rb
from celmech.poisson_series import DFTerm_as_PSterms

m_earth = 3.e-6
masses = np.array([7.4,5.1,8.0,4.8]) * m_earth
resonances = [(4,1),(3,1),(4,1)]

# initialize Poisson series Hamiltonian in resonant variables

sim = get_chain_rebound_sim(resonances,masses)
pham = cm.PoincareHamiltonian(cm.Poincare.from_Simulation(sim))

periods = []
period = 1
periods.append(period)
max_order = 3
sec_terms = cm.disturbing_function.list_secular_terms(2,max_order)
for j,k in resonances:
    period *= j/sp.S(j-k)
    periods.append(period)

# add poisson series terms    
ps_terms = []
for i1,p1 in enumerate(periods):
    for i2,p2 in zip(range(i1+1,len(periods)),periods[i1+1:]):
        pratio = p2/p1
        p,q = int(sp.numer(pratio)),int(sp.numer(pratio) - sp.denom(pratio))
        # resonant terms
        if q <= max_order:
            print(f"adding {p}:{p-q} terms between planets {i1+1} and {i2+1}")
            res_terms_leading = cm.disturbing_function.list_resonance_terms(p,q, min_order=1,max_order = 1)
            res_terms_rest = cm.disturbing_function.list_resonance_terms(p,q,min_order=2, max_order = max_order)
            for term in res_terms_leading:
                kvec,nu=term
                ps_terms+=DFTerm_as_PSterms(pham,i1+1,i2+1,kvec,nu,(0,0))
                ps_terms+=DFTerm_as_PSterms(pham,i1+1,i2+1,kvec,nu,(1,0))
                ps_terms+=DFTerm_as_PSterms(pham,i1+1,i2+1,kvec,nu,(0,1))
            for term in res_terms_rest:
                kvec,nu=term
                ps_terms+=DFTerm_as_PSterms(pham,i1+1,i2+1,kvec,nu,(0,0))
        
        # secular terms
        ps_terms+=DFTerm_as_PSterms(pham,i1+1,i2+1,[0 for _ in range(6)],[0 for _ in range(4)],(1,0))
        ps_terms+=DFTerm_as_PSterms(pham,i1+1,i2+1,[0 for _ in range(6)],[0 for _ in range(4)],(0,1))
        for kvec,nu in sec_terms:
            ps_terms+=DFTerm_as_PSterms(pham,i1+1,i2+1,kvec,nu,(0,0))
hres_pert_series = PoissonSeries.from_PSTerms(ps_terms)
rc_poisson = ResonantChainPoissonSeries(resonances,masses,hres_pert_series)

### set up initial guess ###
Npl = pham.N-1
L0s = np.array([pham.H_params[cm.poincare._get_Lambda0_symbol(i)] for i in range(1,pham.N)])
mtrx = np.zeros((pham.N-1,pham.N-1))
mtrx[0] = rc_poisson.Tmtrx_inv.T[0,:Npl]
for i in range(Npl-1):
    mtrx[i+1,i] = -3 / L0s[i]
    mtrx[i+1,i+1] = 3 /L0s[i+1]
dLambdas = np.linalg.inv(mtrx) @ np.arange(Npl) * 0.002
dP = rc_poisson.Tmtrx_inv.T[:Npl,:Npl] @ dLambdas / L0s[0]
rc_poisson.dK2 = dP[1]
guess = np.zeros(2*(rc_poisson.N_planar + rc_poisson.M))
guess[rc_poisson.N_planar:rc_poisson.N_planar+rc_poisson.M]=np.array([np.pi,-np.pi/2])
guess[-rc_poisson.M:] = dP[2:]
xeq = newton_solve2(rc_poisson.planar_flow_and_jacobian,guess)
f,Df = rc_poisson.planar_flow_and_jacobian(xeq)
eigs = np.linalg.eigvals(Df)
assert np.all(np.isclose(np.real(eigs),0)), "Non-zero real eigenvalue at equilibrium"


