import celmech as cm
import numpy as np
from celmech.poisson_series import DFTerm_as_PSterms
from celmech.disturbing_function import list_resonance_terms
from resonant_chains import resonant_chain_variables_transformation_matrix
from celmech.poisson_series import PoissonSeries, PSTerm
from resonant_chain_utils import Deltas_to_pvars
from scipy.optimize import root
import warnings
def Delta_inner_to_Deltas_3br(Delta_inner,resonances):
    ns = np.zeros(len(resonances) + 1)
    j0,k0 = resonances[0]
    ns[0] = 1
    ns[1] = (j0-k0) * ns[0] / j0 / (1+Delta_inner)
    for i,(rin,rout) in enumerate(zip(resonances,resonances[1:])):
        jo,ko = rout
        ji,ki = rin
        ns[i+2] = (ji * ns[i+1] + (ki - ji) * ns[i] - (ko - jo) * ns[i+1]) / jo
    Deltas = np.array([ (j-k) *ni/no / j - 1 for (j,k),ni,no in zip(resonances,ns,ns[1:]) ])
    return Deltas

from fractions import Fraction
from fractions import Fraction

def get_nonadjacent_first_order_resonances_dict(resonances):
    periods = [Fraction(1, 1)]
    for j, k in resonances:
        periods.append(periods[-1] * Fraction(j, j - k))

    out = {}
    for i1, p1 in enumerate(periods, start=1):
        for i2, p2 in enumerate(periods, start=1):
            r = p2 / p1
            if r.numerator - r.denominator == 1 and i2-i1 > 1:
                out[(i1, i2)] = (r.numerator, 1)
    return out

def resonant_termQ(term,kvecs_3br):
    mtrx = np.concatenate([kvecs_3br,[term.q]])
    return np.linalg.matrix_rank(mtrx) < mtrx.shape[0]

def get_three_br_Hav_series(masses,resonances,Delta_in,
                           jmax_fo = 10,# max first-order index
                           jmax_zo = 10,# max zeroth-order index
                           ):
        Npl = len(resonances) + 1
        kvecs_3br = resonant_chain_variables_transformation_matrix(resonances,max_order=1)[2:Npl,:Npl].astype(int)
        new_q = lambda term : np.round(np.linalg.lstsq(kvecs_3br.T , term.q ,rcond=-1)[0]).astype(int)

        threeBR_filter = lambda term: resonant_termQ(term,kvecs_3br) and np.sum(term.p)<1 and np.sum(term.k)==0 and np.sum(term.kbar)==0
        filter_series = lambda series,filt : cm.poisson_series.PoissonSeries.from_PSTerms([term for term in series.terms if filt(term)])

        Deltas = Delta_inner_to_Deltas_3br(Delta_in,resonances)
        pvars = Deltas_to_pvars(Deltas,resonances,masses)
        pham = cm.PoincareHamiltonian(pvars)
        nonadjacent_first_order_resonances = get_nonadjacent_first_order_resonances_dict(resonances)
        series_terms = []
        
        # first order MMRs
        for m in range(jmax_fo):
            for k,nu in list_resonance_terms(m,1,max_order=1,inclinations=False):
                # add pairwise interactions between adjacent planets
                for pl_indx in range(1,pham.N-1):
                    series_terms += DFTerm_as_PSterms(pham,pl_indx,pl_indx+1,k,nu,(0,0))
                    
        # synodic terms
        for j in range(1,jmax_zo):
            for dl in [(0,0),(0,1),(1,0)]: 
                for pl_indx in range(1,pham.N-1):
                    series_terms += DFTerm_as_PSterms(pham,pl_indx,pl_indx+1,(j, -j, 0, 0, 0, 0), (0, 0, 0, 0),dl)
            
        # non-adjacent pair(s)
        for pair,resonance in nonadjacent_first_order_resonances.items():
            i1,i2 = pair
            for k,nu in list_resonance_terms(*resonance,max_order=1,inclinations=False):
                #add pairwise interactions between adjacent planets
                series_terms += DFTerm_as_PSterms(pham,i1,i2,k,nu,(0,0))
            
        pert_series = cm.poisson_series.PoissonSeries.from_PSTerms(series_terms)
        
        omega_vec = pham.calculate_flow()[:3*Npl:3]
        Domega_vec = np.diag(pham.calculate_jacobian()[:3*Npl:3,pham.N_dof::3])
        chi_series = cm.poisson_series.PoissonSeries_to_GeneratingFunctionSeries(
            pert_series,
            omega_vec,
            Domega_vec
        )
        H2 = 0.5 * chi_series.Lie_deriv(pert_series)
        
        H2av = filter_series(H2,threeBR_filter)
        H2av_transformed = PoissonSeries.from_PSTerms([PSTerm(term.C,[],[],[0] * (Npl - 2),new_q(term)) for term in H2av.terms])
        return H2av_transformed

def get_three_br_chi_series(masses,resonances,Delta_in,
                           jmax_fo = 10,# max first-order index
                           jmax_zo = 10,# max zeroth-order index
                           ):
    Npl = len(resonances) + 1
    kvecs_3br = resonant_chain_variables_transformation_matrix(resonances,max_order=1)[2:Npl,:Npl].astype(int)
    new_q = lambda term : np.round(np.linalg.lstsq(kvecs_3br.T , term.q ,rcond=-1)[0]).astype(int)
    
    threeBR_filter = lambda term: resonant_termQ(term,kvecs_3br) and np.sum(term.p)<1 and np.sum(term.k)==0 and np.sum(term.kbar)==0
    filter_series = lambda series,filt : cm.poisson_series.PoissonSeries.from_PSTerms([term for term in series.terms if filt(term)])
    
    Deltas = Delta_inner_to_Deltas_3br(Delta_in,resonances)
    pvars = Deltas_to_pvars(Deltas,resonances,masses)
    pham = cm.PoincareHamiltonian(pvars)
    nonadjacent_first_order_resonances = get_nonadjacent_first_order_resonances_dict(resonances)
    series_terms = []
    
    # first order MMRs
    for m in range(jmax_fo):
        for k,nu in list_resonance_terms(m,1,max_order=1,inclinations=False):
            # add pairwise interactions between adjacent planets
            for pl_indx in range(1,pham.N-1):
                series_terms += DFTerm_as_PSterms(pham,pl_indx,pl_indx+1,k,nu,(0,0))
                
    # synodic terms
    for j in range(1,jmax_zo):
        for dl in [(0,0),(0,1),(1,0)]: 
            for pl_indx in range(1,pham.N-1):
                series_terms += DFTerm_as_PSterms(pham,pl_indx,pl_indx+1,(j, -j, 0, 0, 0, 0), (0, 0, 0, 0),dl)
        
    # non-adjacent pair(s)
    for pair,resonance in nonadjacent_first_order_resonances.items():
        i1,i2 = pair
        for k,nu in list_resonance_terms(*resonance,max_order=1,inclinations=False):
            #add pairwise interactions between adjacent planets
            series_terms += DFTerm_as_PSterms(pham,i1,i2,k,nu,(0,0))
        
    pert_series = cm.poisson_series.PoissonSeries.from_PSTerms(series_terms)
    
    omega_vec = pham.calculate_flow()[:3*Npl:3]
    Domega_vec = np.diag(pham.calculate_jacobian()[:3*Npl:3,pham.N_dof::3])
    chi_series = cm.poisson_series.PoissonSeries_to_GeneratingFunctionSeries(
        pert_series,
        omega_vec,
        Domega_vec
    )
    return chi_series
def get_three_br_phi_dots(masses,resonances,Delta_in,
                           jmax_fo = 10,# max first-order index
                           jmax_zo = 10,# max zeroth-order index
                        ):
    H2av_transformed = get_three_br_Hav_series(masses,resonances,Delta_in,jmax_fo,jmax_zo)
    Nphi = H2av_transformed.M
    eyeNphi = np.eye(Nphi,dtype = int)
    zeroNphi = np.zeros(Nphi,dtype = int)
    phi_i_dot_list = []
    for i in range(Nphi):
        phi_i_dot = H2av_transformed.Lie_deriv(PSTerm(1e14,[],[],eyeNphi[i],zeroNphi).as_series())
        phi_i_dot_list.append(phi_i_dot)
    return phi_i_dot_list

def get_phi_guess(resonances):
    Nphi = len(resonances) - 2
    T = resonant_chain_variables_transformation_matrix(resonances,max_order=1)
    return np.pi * np.array([kvec[i+2]/j for i,((j,_),kvec) in enumerate(zip(resonances[1:], T[2:]))])

def get_three_br_eq_angles(masses,resonances,Delta_in,
                           phi_guess = None,
                           jmax_fo = 10,# max first-order index
                           jmax_zo = 10,# max zeroth-order index
                        ):
    phi_dots = get_three_br_phi_dots(masses,resonances,Delta_in,jmax_fo,jmax_zo)
    Nphi = len(phi_dots)
    zeroNphi = np.zeros(Nphi)
    f = lambda theta_arr: np.real([phi_dot([],zeroNphi,theta_arr) for phi_dot in phi_dots])
    if phi_guess is None:
        phi_guess = get_phi_guess(resonances)
    rt = root(f,phi_guess)
    if not rt.success:
        warnings.warn(f"Root finding failed: {rt.message}", RuntimeWarning)
    return np.mod(rt.x,2*np.pi)