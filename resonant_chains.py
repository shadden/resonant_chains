import celmech as cm
import numpy as np
from celmech.canonical_transformations import CanonicalTransformation
import sympy as sp
import rebound as rb
from celmech.nbody_simulation_utilities import add_canonical_heliocentric_elements_particle
import warnings

_RT2 = np.sqrt(2)
_RT2_INV = 1/_RT2

def _explicit_mass_dependence_substitution(expression,N):
    L0 = [symbols(r"\Lambda_{{{0}\,0}}".format(i),positive=True) for i in range(1,N)]
    ai0 = [symbols(r"a_{{{0}\,0}}".format(i),positive=True) for i in range(1,N)]
    m = sp.symbols("m(1:{})".format(N))
    mu = sp.symbols("mu(1:{})".format(N))
    M = sp.symbols("M(1:{})".format(N))
    G = sp.symbols("G")
    Mstar = sp.symbols("M")
    L0rule = {L:m*sp.sqrt(G*M_i*a) for L,m,M_i,a in zip(L0,mu,M,ai0)}
    mass_rule = {mu_i:m_i*Mstar/(Mstar + m_i) for m_i,mu_i in zip(m,mu)}
    mass_rule.update({M_i:(Mstar + m_i) for m_i,M_i in zip(m,M)})
    return expression.xreplace(L0rule).xreplace(mass_rule)

def _get_nu_from_resonances(resonances):
    N = len(resonances) - 2
    nu = []
    nu_i = sp.Mul(*[r[0] for r in resonances])
    nu.append(nu_i)
    nu1_denom = 1
    i=0
    for j,k in resonances:
        nu_i *= (j-k)/sp.S(j)
        if i<N-2:
            nu1_denom *= (j-k)
        nu.append(nu_i)
        i+=1
    jN,kN = resonances[-1]
    nu = (jN - kN) * sp.Matrix(nu) / nu1_denom
    return nu


def _swap_entries(list,i1,i2):
    list[i1],list[i2] = list[i2],list[i1]

from celmech.poisson_series import PSTerm,PoissonSeries
def list_multinomial_exponents(pwr,ndim):
    r"""
    List exponents appearing in the expansion of the multinomial 
    
    .. math:: 
    (x_1 + x_2 + ... + x_\mathrm{ndim})^pwr

    Parameters
    ----------
    pwr : int
        Exponent of multinomial
    ndim : int
        Number of variables in multinomial

    Returns
    -------
    list
        A list of lists of length `ndim` representing the exponents of each
        varaible :math:`x_i`
    """
    if ndim==1:
        return [[pwr],]
    else:
        x =[]
        for pow1 in range(0,pwr+1):
            subpows = list_multinomial_exponents(pwr-pow1,ndim-1)
            x+=[[pow1]+y for y in subpows]
        return x    
from math import factorial
def multinomial_coefficient(p, ks):
    """Calculate multinomial coefficient for given p and ks"""
    num = factorial(p)
    denom = 1
    for k in ks:
        denom *= factorial(k)
    return num // denom

def action_exponents_list_to_PSTerms_args(exps_list,N,M):
    I_N=np.eye(N,dtype=int)
    I_M=np.eye(M,dtype=int)
    p_args = np.sum([p_i * I_M[i] for i,p_i in enumerate(exps_list[:M])],axis=1)
    k_args = np.sum([k_i * I_N[i] for i,k_i in enumerate(exps_list[M:])],axis=1)
    return k_args,p_args

def get_Hkep_series_res_variables(pham,Tmtrx,max_order = 2):
    r"""
    Get Keplerian Hamiltonian
    .. math::
        n_i \delta\Lambda_i - \frac{3n_i}{2\Lambda_i} \delta\Lambda_i^2
    as a Poisson series in resonat variables

    Parameters
    ----------
    pham : celmech.poincare.PoincareHamiltonian
        Base Hamiltonian to use in expansion
    Tmtrx : ndarray
        Transformation matrix defining the resonant variables. The matrix should be of the form returned by :func:`resonant_chain_variables_transformation_matrix`.
    max_order : int, optional
        Maximum order of expansion in :math:`\delta\Lambda / \Lambda`
    
    Returns
    -------
    PoissonSeries
        PoissonSeries representation of Hamiltonian
    """
    N_dof = pham.N_dof
    M = N_dof // 3
    N = 2 * M
    zero_M = np.zeros(M,dtype=int)
    ps = pham.particles[1:]
    h0_series = PoissonSeries(N,M)
    for i,row in enumerate(Tmtrx.T[:M]):    
        p = ps[i]
        L0i = pham.H_params[pham.Lambda0s[i+1]]
        for l in range(2,max_order+1):
            prefactor = (-1)**(1-l) * 0.5 * p.n * (1+l) * L0i**(1-l)            
            exps = list_multinomial_exponents(l,N_dof)
            for exp in exps:
                k_args,p_args = action_exponents_list_to_PSTerms_args(exp,N,M)
                C = prefactor*multinomial_coefficient(l,exp)*np.prod(row**exp)
                if C:
                    h0_series+=PSTerm(C,k_args,k_args,p_args,zero_M).as_series()
    return h0_series                   

def transform_poincare_poisson_series(Tmtrx, series):   
    """
    Transform Poisson series in Poincare variables to a Poisson series in resonant variables.

    Parameters
    ----------
    Tmtrx : ndarray
        Transformation matrix defining the resonant variables. The matrix should be of the form returned by :func:`resonant_chain_variables_transformation_matrix`.

    series : celmech.poisson_series.PoissonSeries
        Series to transform

    Returns
    -------
    celmech.poisson_series.PoissonSeries
        Transformed Poisson series

    Raises
    ------
    RuntimeError
        _description_
    """
    Tmtrx_inv = np.linalg.inv(Tmtrx)
    Npl = series.M
    I_N = np.eye(Npl)
    I_2N = np.eye(2*Npl)
    zero_N = np.zeros(Npl)
    new_series_terms = []
    for term in series.terms:
        newq = term.q @ Tmtrx_inv[:Npl,:Npl]
        newq[1] += np.sum(term.k-term.kbar)
        assert np.all(np.isclose(np.mod(newq,1),0))
        newq = np.round(newq).astype(int)
        ptot = np.sum(term.p)
        if ptot==0:
            new_term = PSTerm(term.C,term.k,term.kbar,term.p,newq)
            new_series_terms.append(new_term)
        elif ptot==1:
            i_Lambda = np.argmax(term.p)
            for j,c in enumerate(Tmtrx.T[i_Lambda,:Npl]):
                new_term = PSTerm(c*term.C,term.k,term.kbar,I_N[j],newq)
                new_series_terms.append(new_term)
            for j,c in enumerate(Tmtrx.T[i_Lambda,Npl:2*Npl]):
                new_term = PSTerm(c*term.C,term.k + I_2N[j],term.kbar + I_2N[j],zero_N,newq)
                new_series_terms.append(new_term)
            for j,c in enumerate(Tmtrx.T[i_Lambda,2*Npl:]):
                new_term = PSTerm(c*term.C,term.k + I_2N[Npl+j],term.kbar + I_2N[Npl+j],zero_N,newq)
                new_series_terms.append(new_term)
        else:
            raise RuntimeError("Expansion in Lambdas beyond first order not supported!")
    new_series = PoissonSeries.from_PSTerms(new_series_terms)
    return new_series

    
class ResonantChain():
    def __init__(self,
    resonances, 
    masses,
    tau_alpha,
    tau_e,
    diss_p = 1,
    add_MMR_terms_kwargs = {}, 
    add_secular_terms_kwargs = {}
    ):
        self.N = len(masses) + 1
        self.resonances = resonances
        self.masses = np.array(masses)
        self.tau_alpha = tau_alpha
        self.tau_e = tau_e
        self.diss_p = diss_p
        kam,ct = get_resonant_chain_hamiltonian(
            resonances,
            masses,
            add_MMR_terms_kwargs = add_MMR_terms_kwargs, 
            add_secular_terms_kwargs = add_secular_terms_kwargs
        )
        ham  = cm.hamiltonian.reduce_hamiltonian(kam)
        ham.H = _explicit_mass_dependence_substitution(ham.H,self.N)
        self.ham = ham
        self.ct = ct
        # add Mstar to the parameters of the Hamiltonian... a bit hackish
        Hpars = self.ham.H_params
        self.G = Hpars[sp.symbols("G")]
        self.Mstar= Hpars[sp.symbols("M1")] - Hpars[sp.symbols("m1")]
        self.ham.H_params[sp.symbols("M")] = self.Mstar
        dK1,dK2 = self.ham.full_qp_vars[self.ham.full_N_dof:self.ham.full_N_dof+2]
        kappa1,kappa2 = self.ham.full_qp_vars[:2]
        self._mass_symbols = list(sp.symbols("m(1:{})".format(self.N)))
        self.ham.H_params[dK1] = 0.0
        self.Npars = self.ham.H_params.copy()
        for key in self._mass_symbols:
            self.Npars.pop(key,None)
        self.Npars.pop(dK2)
        self.Nflow = self.ham.flow.xreplace(self.Npars)
        self.Njac = self.ham.jacobian.xreplace(self.Npars)

        self.arg_symbols = self.ham.qp_vars + [dK2] + self._mass_symbols
        self._arg_symbols_x1_dK2_swap = self.arg_symbols.copy()
        self.i_dK2 = self.arg_symbols.index(dK2)
        self.i_x1 = self.arg_symbols.index(sp.symbols("x1",real=True))
        _swap_entries(self._arg_symbols_x1_dK2_swap,self.i_x1,self.i_dK2)
        self._flow_func_x1_par = sp.lambdify(self._arg_symbols_x1_dK2_swap,self.Nflow)
        self._jac_func_x1_par = sp.lambdify(
            self._arg_symbols_x1_dK2_swap,
            sp.Matrix(
                self.ham.N_dim,self.ham.N_dim,
                lambda i,j: sp.diff(self.Nflow[i], self._arg_symbols_x1_dK2_swap[j])
            )
        )
        self._flow_func = sp.lambdify(self.arg_symbols,self.Nflow)
        self._jac_func = sp.lambdify(self.arg_symbols,self.Njac)
        self._initialize_ct(ct)

        dkappa1_dt = sp.diff(self.ham.H,dK1).xreplace(self.Npars)
        dkappa2_dt = sp.diff(self.ham.H,dK2).xreplace(self.Npars)
        self._kappa1_dot_func = sp.lambdify(self.arg_symbols,dkappa1_dt)
        self._kappa2_dot_func = sp.lambdify(self.arg_symbols,dkappa2_dt)

        ########################
        # Dissipative dynamics #
        ########################
        dis_vec,tau_e_symbols,tau_alpha_symbols,p_symbol = self._construct_dissipation_flow()
        full_flow = sp.Matrix(list(self.Nflow) + [0]) + dis_vec
        Nfull_flow = full_flow.xreplace(self.Npars)
        self.full_symbols = self.arg_symbols + tau_alpha_symbols + tau_e_symbols + [p_symbol]
        self._full_flow_func = sp.lambdify(self.full_symbols,Nfull_flow)
        dyvars = self.full_symbols[:self.ham.N_dim+1]
        Nfull_jac = sp.Matrix(self.ham.N_dim+1,self.ham.N_dim+1,lambda i,j: sp.diff(Nfull_flow[i],dyvars[j]))
        self._full_jac_func = sp.lambdify(self.full_symbols,Nfull_jac)

    def f_dis(self,y):
        return self._full_flow_func(*y,*self.masses,*self.tau_alpha,*self.tau_e,self.diss_p).reshape(-1)
    def Df_dis(self,y):
        return self._full_jac_func(*y,*self.masses,*self.tau_alpha,*self.tau_e,self.diss_p)

    def _initialize_ct(self,ct):
        canonical_vars_matrix = sp.Matrix([
            sp.symbols("lambda{0},eta{0},rho{0},Lambda{0},kappa{0},sigma{0}".format(i)) for i in range(1,self.N)
        ])
        canonical_vars_matrix = ct.old_to_new(canonical_vars_matrix)
        canonical_vars_matrix = _explicit_mass_dependence_substitution(canonical_vars_matrix,self.N)
        canonical_vars_matrix = canonical_vars_matrix.xreplace(self.Npars)
        self._canonical_vars_matrix_func = sp.lambdify(self.arg_symbols,canonical_vars_matrix)
    def kappa1_dot(self,x,dK2val):
        return self._kappa1_dot_func(*x,dK2val,*self.masses)
    def kappa2_dot(self,x,dK2val):
        return self._kappa2_dot_func(*x,dK2val,*self.masses)
    def f(self,x,dK2val):
        return self._flow_func(*x,dK2val,*self.masses).reshape(-1)
    def Df(self,x,dK2val):
        return self._jac_func(*x,dK2val,*self.masses)
    def solve_eq_x1(self,x1,guess = None,**kwargs):
        if guess is None:
            guess = np.zeros(self.ham.N_dim)
        f = lambda x: self._flow_func_x1_par(*x,x1,*self.masses).reshape(-1)
        Df = lambda x: self._jac_func_x1_par(*x,x1,*self.masses)
        soln = newton_solve(f,Df,guess,**kwargs)
        dK2 = soln[self.i_x1]
        soln[self.i_x1] = x1
        return soln, dK2
    def get_Poincare(self,x,dK2val):
        cvars_matrix = self._canonical_vars_matrix_func(*x,dK2val,*self.masses)
        particles = []
        varnames = "l,eta,rho,Lambda,kappa,sigma".split(",")
        for m,cvars in zip(self.masses,cvars_matrix): 
            kwargs = dict(zip(varnames,cvars))
            particle = cm.poincare.PoincareParticle(G=self.G,m=m,Mstar=self.Mstar,**kwargs)
            particles.append(particle)
        pvars = cm.Poincare(self.G,particles)
        return pvars
    def _construct_dissipation_flow(self):

        p = sp.symbols("p")
        N_Phi = self.N - 3
        tau_e = [cm.get_symbol(r"\tau_{{e\,{}}}".format(i)) for i in range(1,self.N)]
        tau_alpha = [cm.get_symbol(r"\tau_{{\alpha\,{}}}".format(i)) for i in range(2,self.N)]
        L10 = cm.poincare._get_Lambda0_symbol(1)    
        Lambda = self.ct.old_qp_vars[3*(self.N-1)::3]
        rho = [cm.poincare._get_Lambda0_symbol(i)/cm.poincare._get_Lambda0_symbol(1) for i in range(1,self.N)]
        Gamma = [(self.ham.qp_vars[self.N-3+i]**2+ self.ham.qp_vars[self.ham.N_dof+self.N-3+i]**2)/2 for i in range(self.N-1)]
        e_sq = [2* G  / r for G,L,r in zip(Gamma,Lambda,rho)]
        gamma_e = [1/te for te in tau_e]
        gamma_alpha =  [1/ta for ta in tau_alpha]
        gamma_a = [2 * p * e_sq[0] * gamma_e[0] ] + [ g_alpha  +  2 * p * e2 * ge  for g_alpha,e2,ge in zip(gamma_alpha,e_sq[1:],gamma_e[1:])]
        mtrx_fn = lambda i,j: sp.diff(get_Phi_i_of_Lambda(i),Lambda[j]) * L10
        get_Phi_i_of_Lambda = lambda i: self.ct.new_to_old(self.ham.full_qp_vars[self.ham.full_N_dof+2+i])
        dPhi_i_dLambda_j = sp.Matrix(N_Phi,self.N-1,mtrx_fn)
        nu = _get_nu_from_resonances(self.resonances)
        denom=0
        dK2_dot_dis = 0
        dPhi_dot_dis = sp.Matrix([0 for _ in range(N_Phi)])
        for i in range(0,self.N-1):
            denom += nu[i] * rho[i]
            for l in range(i+1,self.N-1):
                dK2_dot_dis += (nu[l] - nu[i]) * rho[i] * rho[l] * (gamma_a[l] - gamma_a[i]) / 2 
                for j in range(N_Phi):
                    dPhi_dot_dis[j] += (nu[l] * dPhi_i_dLambda_j[j,i] - nu[i] * dPhi_i_dLambda_j[j,l]) * rho[i] * rho[l] * (gamma_a[l] - gamma_a[i]) / 2
        dK2_dot_dis /= denom 
        dPhi_dot_dis /= denom
        for i,ge in enumerate(gamma_e):
            dK2_dot_dis += 2 * ge * Gamma[i] 

        dis_vec = sp.Matrix([0 for _ in range(self.ham.N_dim+1)])
        
        dis_vec[-1] = dK2_dot_dis
        for i,dPhi_dot in enumerate(dPhi_dot_dis):
            dis_vec[self.ham.N_dof+i] = dPhi_dot
            
        for i in range(self.N-1):
            dis_vec[self.N-3+i] = -gamma_e[i] * self.ham.qp_vars[self.N-3+i]
            dis_vec[self.ham.N_dof + self.N-3+i] = -gamma_e[i] * self.ham.qp_vars[self.ham.N_dof + self.N-3+i]
                            
        return _explicit_mass_dependence_substitution(dis_vec,self.N),tau_e,tau_alpha,p

def get_chain_rebound_sim(resonances, masses):
    sim = rb.Simulation()
    sim.add(m=1)
    Period = 1
    add_canonical_heliocentric_elements_particle(masses[0],{'a' : Period**(2/3)},sim)
    for mass,pq in zip(masses[1:],resonances):
        p,q = pq
        Period *= p/(p-q)
        add_canonical_heliocentric_elements_particle(mass,{'a' : Period**(2/3)},sim)
    sim.move_to_com()
    return sim

def transform_poincare_poisson_series(Tmtrx, series):   
    """
    Transform Poisson series in Poincare variables to a Poisson series in resonant variables.

    Parameters
    ----------
    Tmtrx : ndarray
        Transformation matrix defining the resonant variables. The matrix should be of the form returned by :func:`resonant_chain_variables_transformation_matrix`.

    series : celmech.poisson_series.PoissonSeries
        Series to transform

    Returns
    -------
    celmech.poisson_series.PoissonSeries
        Transformed Poisson series

    Raises
    ------
    RuntimeError
        _description_
    """
    Tmtrx_inv = np.linalg.inv(Tmtrx)
    Npl = series.M
    I_N = np.eye(Npl)
    I_2N = np.eye(2*Npl)
    zero_N = np.zeros(Npl)
    new_series_terms = []
    for term in series.terms:
        newq = term.q @ Tmtrx_inv[:Npl,:Npl]
        newq[1] += np.sum(term.k-term.kbar)
        newq = np.round(newq).astype(int)
        ptot = np.sum(term.p)
        if ptot==0:
            new_term = PSTerm(term.C,term.k,term.kbar,term.p,newq)
            new_series_terms.append(new_term)
        elif ptot==1:
            i_Lambda = np.argmax(term.p)
            for j,c in enumerate(Tmtrx.T[i_Lambda,:Npl]):
                new_term = PSTerm(c*term.C,term.k,term.kbar,I_N[j],newq)
                new_series_terms.append(new_term)
            for j,c in enumerate(Tmtrx.T[i_Lambda,Npl:2*Npl]):
                new_term = PSTerm(c*term.C,term.k + I_2N[j],term.kbar + I_2N[j],zero_N,newq)
                new_series_terms.append(new_term)
            for j,c in enumerate(Tmtrx.T[i_Lambda,2*Npl:]):
                new_term = PSTerm(c*term.C,term.k + I_2N[Npl+j],term.kbar + I_2N[Npl+j],zero_N,newq)
                new_series_terms.append(new_term)
        else:
            raise RuntimeError("Expansion in Lambdas beyond first order not supported!")
    new_series = PoissonSeries.from_PSTerms(new_series_terms)
    return new_series
    
def get_lambda_kvecs_in_df(resonances,max_order = None):
    Npl = len(resonances)+1
    P = 1
    Periods = [P,]
    for i,jk in enumerate(resonances):
        j,k = jk
        P *= j / sp.S(j-k)
        Periods.append(P)
    k_vecs_in_df = []
    max_order = max_order if max_order else np.inf
    for i1 in range(Npl):
        Pin = Periods[i1]
        for i2 in range(i1+1,Npl):
            row = np.zeros(Npl,dtype = int)
            Pout = Periods[i2]
            ratio = Pout/Pin
            j = sp.numer(ratio)
            k = j-sp.denom(ratio)
            if k<=max_order:
                row[i1] = k-j
                row[i2] = j
                k_vecs_in_df.append(row)
    return sp.Matrix(k_vecs_in_df)
            
def resonant_chain_variables_transformation_matrix(resonances,max_order = None):
    # number of planets
    N = len(resonances) + 1
    A = sp.Matrix(np.zeros((3*N,3*N),dtype = int))
    plast,qlast = resonances[-1]
    A[0,N-2] =  1 
    A[0,N-1] = -1
    A[1,N-2] = (qlast - plast) / sp.S(qlast)
    A[1,N-1] = plast / sp.S(qlast)
    for i in range(N-2):
        resvec1_numerators,resvec2_numerators = np.zeros((2,N),dtype=int)
        p1,q1 = resonances[i]
        p2,q2 = resonances[i+1]
        resvec1_numerators[i] = (q1 - p1)
        resvec1_numerators[i+1] = p1
        resvec2_numerators[i+1] = (q2 - p2)
        resvec2_numerators[i+2] = p2
        row = q1 * resvec2_numerators - q2 * resvec1_numerators
        row = row.astype(int)//np.gcd.reduce(row.astype(int))
        for j in range(N):
            A[i+2,j] = row[j] #// np.gcd.reduce(row)
            A[N+j,N+j] =  1
            A[2*N+j,2*N+j] =  1
            A[N+j,N-2] = (qlast - plast) / sp.S(qlast)
            A[N+j,N-1] = plast / sp.S(qlast)
            A[2*N+j,N-2] = (qlast - plast) / sp.S(qlast)
            A[2*N+j,N-1] = plast / sp.S(qlast)
    A_inv = A.inv()
    k_vecs_in_df = get_lambda_kvecs_in_df(resonances,max_order = max_order)
    C = sp.Matrix(np.eye(3*N,3*N,dtype=int))
    new_kvecs = k_vecs_in_df * A_inv[:N,:N]
    for l in range(N-2):
        denoms = [sp.denom(u) for u in new_kvecs[:,2+l]]
        nums = [sp.numer(u) for u in new_kvecs[:,2+l]]
        c_inv = np.lcm.reduce(denoms) / sp.S(np.gcd.reduce(nums))
        C[2+l,2+l] = 1/c_inv
    return np.array(C*A,dtype = float) 

from sympy import symbols

def get_resonant_chain_new_variables(Npl):
    # angle variables
    kappa = sp.symbols("kappa(1:3)",real=True)
    phi = symbols('phi(1:{})'.format(Npl-1))
    s = symbols('s(1:{})'.format(Npl+1))
    psi = symbols('psi(1:{})'.format(Npl+1))

    # action-like variables
    K = sp.symbols("K(1:3)",real=True)
    Phi = symbols('Phi(1:{})'.format(Npl-1))
    D = symbols('D(1:{})'.format(Npl+1))
    Psi = symbols('Psi(1:{})'.format(Npl+1))

    new_qp_pairs = list(zip(list(kappa)+list(phi)+list(s)+list(psi),list(K)+list(Phi)+list(D)+list(Psi)))
    return new_qp_pairs
from sympy.simplify.fu import TR8
from celmech import get_symbol
def get_resonant_chain_hamiltonian(resonances, masses,add_MMR_terms_kwargs = {}, add_secular_terms_kwargs = {}):
    sim = get_chain_rebound_sim(resonances,masses)
    pvars = cm.Poincare.from_Simulation(sim)
    pham = cm.PoincareHamiltonian(pvars)
    res_var_mtrx = resonant_chain_variables_transformation_matrix(resonances)
    periods = []
    period = 1
    periods.append(period)
    for j,k in resonances:
        period *= j/sp.S(j-k)
        periods.append(period)
    for i1,p1 in enumerate(periods):
        for i2,p2 in zip(range(i1+1,len(periods)),periods[i1+1:]):
            pratio = p2/p1
            p,q = int(sp.numer(pratio)),int(sp.numer(pratio) - sp.denom(pratio))
            if q <= add_MMR_terms_kwargs['max_order']:
                pham.add_MMR_terms(p,q,indexIn=i1+1,indexOut=i2+1,**add_MMR_terms_kwargs)

    # for i,pq in enumerate(resonances):
    #     pham.add_MMR_terms(*pq,indexIn=i+1,indexOut=i+2,**add_MMR_terms_kwargs)
    for i in range(1,sim.N):
        for j in range(i+1,sim.N):
            pham.add_secular_terms(indexIn=i,indexOut=j,**add_secular_terms_kwargs)
            # shifts of mean motions to order ~mu e^0
            pham.add_cosine_term([0,0,0,0,0,0],l_max=1,indexIn=i,indexOut=j)
    
    ct_res = CanonicalTransformation.from_poincare_angles_matrix(
        pvars,
        res_var_mtrx,
        new_qp_pairs=get_resonant_chain_new_variables(sim.N-1)
    )
    L0s = pham.Lambda0s
    kam = kam = ct_res.old_to_new_hamiltonian(pham)
    kam.H = kam.H.func(*[sp.simplify(TR8(a)) for a in kam.H.args])
    ct_rescale = CanonicalTransformation.rescale_transformation(
        kam.qp_pairs,
        1/L0s[1],
        params = {L0s[1]:pham.H_params[L0s[1]]}
    )
    kam_rescaled = ct_rescale.old_to_new_hamiltonian(kam)
    x = sp.symbols("x(1:{})".format(sim.N),real = True)
    y = sp.symbols("y(1:{})".format(sim.N),real = True)
    r = sp.symbols("r(1:{})".format(sim.N),real = True)
    s = sp.symbols("s(1:{})".format(sim.N),real = True)
    ct_polar2cart = CanonicalTransformation.polar_to_cartesian(
        kam_rescaled.qp_vars,
        [i for i in range(sim.N-1 ,3*(sim.N-1))],
        cartesian_symbol_pairs=list(zip(x,y)) + list(zip(r,s))
    )
    kam_xy = ct_polar2cart.old_to_new_hamiltonian(kam_rescaled)
    
    # from actions to delta-actions
    old_actions = [kam_xy.qp_vars[i] for i in range(kam_xy.N_dof,kam_xy.N_dof + sim.N-1)]
    vec = sp.Matrix(L0s[1:] + [0 for _ in range(2*(sim.N-1))]) / L0s[1]
    action_refs = sp.Matrix(res_var_mtrx).inv().T * vec
    
    ct_delta_action = CanonicalTransformation.actions_to_delta_actions(
        kam_xy.qp_vars,
        actions = old_actions,
        delta_actions=[get_symbol(r'δ' + sp.latex(v)) for v in old_actions],
        actions_ref=action_refs,
        params = {L0:pham.H_params[L0] for L0 in L0s[1:]}
    )
    kam_xy_delta = ct_delta_action.old_to_new_hamiltonian(kam_xy)
    ct_composite = CanonicalTransformation.composite([ct_res,ct_rescale,ct_polar2cart,ct_delta_action])
    
    return kam_xy_delta,ct_composite

from scipy.linalg import solve as lin_solve
def newton_solve(fun,Dfun,guess,params=(),max_iter=100,rtol=1e-6,atol=1e-12):
    y = guess.copy()
    for itr in range(max_iter):
        f = fun(y,*params)
        Df = Dfun(y,*params)
        dy = -1 * lin_solve(Df,f)
        y+=dy
        if np.all( np.abs(dy) < rtol * np.abs(y) + atol ):
            break
    else:
        warnings.warn("did not converge")
    return y
def newton_solve2(fun_and_Dfun,guess,params=(),max_iter=100,rtol=1e-6,atol=1e-12):
    y = guess.copy()
    for itr in range(max_iter):
        f,Df = fun_and_Dfun(y,*params)
        dy = -1 * lin_solve(Df,f)
        y+=dy
        if np.all( np.abs(dy) < rtol * np.abs(y) + atol ):
            break
    else:
        warnings.warn("did not converge")
    return y


def tau_alphas_to_tau_as(tau_alpha,masses,resonances):
    Npl = len(masses)
    sma = np.ones(Npl)
    mtrx = -1 * np.eye(Npl)
    mtrx[:,0] = 1
    for i,jk in enumerate(resonances):
        j,k = jk
        sma[i+1] = sma[i] * (j/(j-k))**(2/3)
    mtrx[0] = masses/sma
    gamma_alphas = np.concatenate(([0],1/np.array(tau_alpha)))
    gamma_a = np.linalg.inv(mtrx) @ gamma_alphas
    return 1/gamma_a

def hamiltonian_series_to_flow_series_list(ham_series):
    r"""
    Generate a list of Poisson series that can be evaluated to get the flow generated by the input Hamiltonian

    Parameters
    ----------
    ham_series : celmech.poisson_series.PoissonSeries
        Poisson series of the input Hamiltonian

    Returns
    -------
    list
        A list of :code:`celmech.poisson_series.PoissonSeries` objects giving the flow generated by the Hamiltonian.
        The list entries are:
        .. math::
            \begin{pmatrix}
                \frac{d}{dt}z_1
                \\
                \vdots
                \\
                \frac{d}{dt}z_N 
                \\
                \frac{d}{dt}P_1 
                \\
                \vdots
                \\
                \frac{d}{dt}P_M
                \\
                \frac{d}{dt}Q_1
                \\
                \vdots
                \\
                \frac{d}{dt}Q_M
            \end{pmatrix}
    """
    I_N = np.eye(ham_series.N,dtype=int)
    zero_N = np.zeros(ham_series.N,dtype=int)
    I_M = np.eye(ham_series.M,dtype=int)
    zero_M = np.zeros(ham_series.M,dtype=int)
    flow_series_list = []
    # dx/dt
    for i in range(ham_series.N):
        var_series = PSTerm(1,I_N[i],zero_N,zero_M,zero_M).as_series()
        dxi_dt = ham_series.Lie_deriv(var_series)
        flow_series_list.append(dxi_dt)
        
    # dP/dt 
    for i in range(ham_series.M):
        var_series = PSTerm(1,zero_N,zero_N,I_M[i],zero_M).as_series()
        dPi_dt = ham_series.Lie_deriv(var_series)
        flow_series_list.append(dPi_dt)
    
    # dQ/dt 
    for i in range(ham_series.M):
        var_series = PSTerm(1,zero_N,zero_N,zero_M,I_M[i]).as_series()
        dexp_iQ_dt = ham_series.Lie_deriv(var_series)
        factor = PSTerm(-1j,zero_N,zero_N,zero_M,-I_M[i]).as_series()
        dQi_dt = factor * dexp_iQ_dt
        flow_series_list.append(dQi_dt)
    return flow_series_list

def dseries_dQi(series,i):
    return PoissonSeries.from_PSTerms([1j * term.q[i] * term for term in series.terms if term.q[i]], N=series.N, M=series.M)
def dseries_dPi(series,i):
    one_i = np.eye(series.M)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.p[i], term.k,term.kbar,term.p - one_i,term.q) for term in series.terms if term.p[i]], N=series.N, M=series.M)
def dseries_dzi(series,i):
    one_i = np.eye(series.N)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.k[i], term.k - one_i,term.kbar,term.p,term.q) for term in series.terms if term.k[i]], N=series.N, M=series.M)
def dseries_dzbari(series,i):
    one_i = np.eye(series.N)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.kbar[i], term.k,term.kbar - one_i,term.p,term.q) for term in series.terms if term.kbar[i]], N=series.N, M=series.M)

def real_vars_jacobian_series(complex_flow_series):
    flow0 = complex_flow_series[0]
    N,M = flow0.N,flow0.M
    N_dim = 2 * (N+M)
    jac = [[None for j in range(N_dim)] for i in range(N+2*M)]
    for i,flow_i in enumerate(complex_flow_series):
        for j in range(N):
            df_dz = dseries_dzi(flow_i,j)
            df_dzbar = dseries_dzbari(flow_i,j)
            df_dy = 1j * _RT2_INV * (df_dzbar + -1*df_dz)
            df_dx = _RT2_INV * (df_dzbar + df_dz)
            jac[i][j] = df_dy
            jac[i][N+M+j] = df_dx
        for j in range(M):            
            df_dP = dseries_dPi(flow_i,j)
            df_dQ = dseries_dQi(flow_i,j)
            jac[i][N+j] = df_dQ
            jac[i][2*N+M+j] = df_dP
    return jac

def _real_flow_and_jacobian(X,complex_flow_list,jacobian_series,N,M):
    y = X[:N]
    Q = X[N:N+M]
    x = X[N+M:2*N + M]
    P = X[2*N + M:2*N + 2*M]
    z = np.sqrt(0.5) * (x - 1j * y)
    zdot = np.array([f(z,P,Q) for f in complex_flow_list[:N]])
    Pdot = np.real([f(z,P,Q) for f in complex_flow_list[N:N+M]])
    Qdot = np.real([f(z,P,Q) for f in complex_flow_list[N+M:]])
    jj = np.array([[series(z,P,Q) for series in row] for row in jacobian_series])
    dzdot_dvar = jj[:N,:]
    dxdot_dvar = _RT2 * np.real(dzdot_dvar)
    dydot_dvar = -_RT2 * np.imag(dzdot_dvar)
    dPdot_dvar = np.real(jj[N:N+M,:])
    dQdot_dvar = np.real(jj[N+M:,:])
    return np.concatenate((-np.imag(zdot)*_RT2,Qdot,np.real(zdot)*_RT2,Pdot)),np.vstack([dydot_dvar,dQdot_dvar,dxdot_dvar,dPdot_dvar])

def _real_flow(X,complex_flow_list,N,M):
    y = X[:N]
    Q = X[N:N+M]
    x = X[N+M:2*N + M]
    P = X[2*N + M:2*N + 2*M]
    z = np.sqrt(0.5) * (x - 1j * y)
    zdot = np.array([f(z,P,Q) for f in complex_flow_list[:N]])
    Pdot = np.real([f(z,P,Q) for f in complex_flow_list[N:N+M]])
    Qdot = np.real([f(z,P,Q) for f in complex_flow_list[N+M:]])
    return np.concatenate((-np.imag(zdot)*_RT2,Qdot,np.real(zdot)*_RT2,Pdot))

def _real_hamiltonian(X,complex_hamiltonian,N,M):
    y = X[:N]
    Q = X[N:N+M]
    x = X[N+M:2*N + M]
    P = X[2*N + M:2*N + 2*M]
    z = np.sqrt(0.5) * (x - 1j * y)
    return complex_hamiltonian(z,P,Q)

def _real_jacobian(X,jacobian_series,N,M):
    y = X[:N]
    Q = X[N:N+M]
    x = X[N+M:2*N + M]
    P = X[2*N + M:2*N + 2*M]
    z = np.sqrt(0.5) * (x - 1j * y)
    jj = np.array([[series(z,P,Q) for series in row] for row in jacobian_series])
    dzdot_dvar = jj[:N,:]
    dxdot_dvar = _RT2 * np.real(dzdot_dvar)
    dydot_dvar = -_RT2 * np.imag(dzdot_dvar)
    dPdot_dvar = np.real(jj[N:N+M,:])
    dQdot_dvar = np.real(jj[N+M:,:])
    return np.vstack([dydot_dvar,dQdot_dvar,dxdot_dvar,dPdot_dvar])
def _Lambda_comb_diss_deriv(a,b,gamma,rho):
    r"""
    Compute the time derivative of (a.Lambda)/(b.Lambda) assuming
    dLambda/dt = -(1/2)*gamma*Lambda. rho=Lambda_i/Lambda_1.
    """
    n = len(a)
    result = 0
    for i in range(n):
        for j in range(i+1,n):
            result += -0.5 * (a[i] * b[j] - a[j] * b[i]) * rho[i] * rho[j] * (gamma[i] - gamma[j])
    result /= (b @ rho)**2
    return result
               
class ResonantChainPoissonSeries():
    def __init__(self,resonances,masses,hpert_series,dK2=0,action_scale = None,h0_order = 2,max_order = None):
        pham = cm.PoincareHamiltonian(cm.Poincare.from_Simulation(get_chain_rebound_sim(resonances,masses)))
        self.pham = pham
        self.Lambda0s = np.array([self.pham.H_params[L0] for L0 in self.pham.Lambda0s[1:]])
        self.Tmtrx = resonant_chain_variables_transformation_matrix(resonances,max_order=max_order)
        self.Tmtrx_inv = np.linalg.inv(self.Tmtrx)
        h_kep = get_Hkep_series_res_variables(pham,self.Tmtrx,max_order=h0_order)
        self.h_full_series  = h_kep + transform_poincare_poisson_series(self.Tmtrx,hpert_series)
        self.complex_full_flow_list = hamiltonian_series_to_flow_series_list(self.h_full_series)
        self.full_jacobian_list = real_vars_jacobian_series(self.complex_full_flow_list)
        self.N_full = self.h_full_series.N
        self.M_full = self.h_full_series.M
        if not action_scale:
            self._action_scale = pham.H_params[pham.Lambda0s[1]]
        self.dK2 = dK2

        # singly-reduced planar Hamiltonian
        # i.e., dependence on dK1 eliminated, dependence on dK2 retained.

        # calculate degree of term in powers of action variables
        term_deg = lambda term: np.sum(term.p) + 0.5*np.sum(term.k + term.kbar)

        full_planar_terms = [
            PSTerm(
                term.C * self.action_scale**(term_deg(term)-1),
                term.k[:self.N_planar],
                term.kbar[:self.N_planar],
                term.p[1:],
                term.q[1:]
            )
            for term in self.h_full_series.terms
            if np.all(term.k[self.N_planar:]==0) and np.all(term.kbar[self.N_planar:]==0) and term.p[0]==0
        ]
        h_full_planar_series = PoissonSeries.from_PSTerms(full_planar_terms)
        assert np.all([term.q[0]==0 for term in h_full_planar_series.terms])
        self.planar_full_complex_flow_list = hamiltonian_series_to_flow_series_list(h_full_planar_series)
        self.planar_full_jacobian_list = real_vars_jacobian_series(self.planar_full_complex_flow_list)

    @property
    def action_scale(self):
        return self._action_scale
    
    @action_scale.setter
    def action_scale(self,val):
        self._action_scale = val
        self._reduce_series

    @property
    def dK2(self):
        return self._dK2 / self.action_scale
    
    @dK2.setter
    def dK2(self,val):
        self._dK2 = val * self.action_scale
        self._reduce_series()

    def _reduce_series(self):
        # Eliminate dependence on conserved quantities K1, K2        
        mask = np.ones(self.M_full,dtype=bool)
        mask[0] = False
        mask[1] = False
        vals = np.array([0,self.dK2])

        # calculate degree of term in powers of action variables
        term_deg = lambda term: np.sum(term.p) + 0.5*np.sum(term.k + term.kbar)

        new_terms = []
        for term in self.h_full_series.terms:
            factor = np.prod(vals**term.p[np.logical_not(mask)]) * self.action_scale**(term_deg(term)-1)
            if factor:
                new_term = PSTerm(
                    term.C * factor ,
                    term.k,
                    term.kbar,
                    term.p[mask],
                    term.q[mask]
                )
                new_terms.append(new_term)            
        kam_series_reduced = PoissonSeries.from_PSTerms(new_terms)
        # reduced hamiltonian
        self.h_series = kam_series_reduced
        self.complex_flow_list = hamiltonian_series_to_flow_series_list(self.h_series)
        self.jacobian_list = real_vars_jacobian_series(self.complex_flow_list)

        # reduced planar hamiltonian
        N_planar = self.h_full_series.N // 2
        planar_terms = [
            PSTerm(
                term.C,
                term.k[:N_planar],
                term.kbar[:N_planar],
                term.p,
                term.q
            )
            for term in self.h_series.terms 
            if np.alltrue(term.k[N_planar:]==0) and np.alltrue(term.kbar[N_planar:]==0)
        ]
        self.h_planar_series = PoissonSeries.from_PSTerms(planar_terms)
        self.planar_complex_flow_list = hamiltonian_series_to_flow_series_list(self.h_planar_series)
        self.planar_jacobian_list = real_vars_jacobian_series(self.planar_complex_flow_list)
        

    @property
    def N(self):
        return self.h_series.N
    @property
    def N_planar(self):
        return self.h_planar_series.N
    @property
    def M(self):
        return self.h_series.M
    
    def planar_dissipation_flow(self,X,tau_e,tau_m,p=0):
        #
        # X = y_1,...,y_N,kappa_2,phi_1,...,phi_M,x_1,...,x_N,dK2,Phi_1,...,Phi_M
        #
        # dlna/dt = -1/tau_m - 2 p e^2/tau_e
        Ndim = 2*(self.M+self.N_planar+1)
        assert X.size==Ndim, "Input 'X' has dimension ({},) but should be dimension ({},)".format(X.size,Ndim)
        Npl = self.N_planar
        rho = self.Lambda0s/self.Lambda0s[0]
        e_sq = np.array([X[i]**2 + X[i+Npl+self.M+1]**2   for i in range(Npl)])/rho

        dis_vec = np.zeros(Ndim)
        gamma_a = 1/tau_m + 2 * p * e_sq / tau_e
    
        Tinv_transpose = self.Tmtrx_inv.T
        
        nu_vec = Tinv_transpose[0,:Npl]
        K1_by_L10 = nu_vec @ rho
        K1dot_by_L10 = -0.5 * nu_vec @ (gamma_a * rho)
        
        for i,a_i in enumerate(Tinv_transpose[1:self.M+2,:Npl]):
            dPdt = a_i @ (-0.5*rho*gamma_a)/ K1_by_L10 - (a_i @ rho) * K1dot_by_L10 / (K1_by_L10**2)
            dis_vec[2*self.N_planar + self.M + 1+i] = dPdt
        dis_vec *= nu_vec @ rho

        for i in range(Npl):
            gamma_e = 1/tau_e[i]
            dis_vec[i] = -1 * gamma_e * X[i]
            dis_vec[i+Npl+self.M+1] = -1 * gamma_e * X[i+Npl+self.M+1]
            # dK2
            dis_vec[2*Npl + self.M+1 ] +=  gamma_e * (X[i]**2 + X[i+Npl+self.M+1]**2)
        return dis_vec

    def planar_dissipation_jacobian(self,X,tau_e,tau_m,p=0):
        #
        # X = y_1,...,y_N,kappa_2,phi_1,...,phi_M,x_1,...,x_N,dK2,Phi_1,...,Phi_M
        #
        # dlna/dt = -1/tau_m - 2 p e^2/tau_e
        Ndim = 2*(self.M+self.N_planar+1)
        assert X.size==Ndim, "Input 'X' has dimension ({},) but should be dimension ({},)".format(X.size,Ndim)
        Npl = self.N_planar
        rho = self.Lambda0s/self.Lambda0s[0]
        dis_jac = np.zeros((Ndim,Ndim))
        D_gamma_a = np.zeros((Npl,Ndim))

        Tinv_transpose = self.Tmtrx_inv.T

        nu_vec = Tinv_transpose[0,:Npl]
        for i in range(Npl):
            i_x = i+Npl+self.M+1
            D_gamma_a[i,i] = 4 * p * X[i] / tau_e[i] / rho[i]
            D_gamma_a[i,i_x] = 4 * p * X[i_x] / tau_e[i] / rho[i]

        K1_by_L10 = nu_vec @ rho
        D_K1_by_L10 = -0.5 * (rho * nu_vec) @ D_gamma_a

        for i,a_i in enumerate(Tinv_transpose[1:self.M+2,:Npl]):    
            i_P = 2 * Npl + self.M + 1 + i
            dis_jac[i_P] += -0.5 * rho * a_i @ D_gamma_a / K1_by_L10
            dis_jac[i_P] += -1 * (a_i @ rho) * D_K1_by_L10 / (K1_by_L10**2)
        dis_jac *= nu_vec @ rho

        i_dK2 = 2 * Npl + self.M+1 
        for i in range(Npl):
            gamma_e = 1/tau_e[i]
            dis_jac[i,i] += -1 * gamma_e
            i_x = i+Npl+self.M+1
            dis_jac[i_x,i_x] += -1 * gamma_e
            # dK2
            dis_jac[i_dK2,i] += 2 * gamma_e * X[i]
            dis_jac[i_dK2,i_x] += 2 * gamma_e * X[i_x]
        return dis_jac

    def planar_flow_with_dissipation(self,X,tau_e,tau_a,p):
        shape = 2*(self.N_planar+self.M) + 1
        msg = "Input X has dimensions {} but should be size ({},)".format(X.shape,shape)
        assert X.size==2*(self.N_planar+self.M) + 1, msg
        i_kappa2 = self.N_planar
        X_full = np.insert(X,i_kappa2,0.)
        f_cons = self.planar_full_flow(X_full)
        f_dis = self.planar_dissipation_flow(X_full,tau_e,tau_a,p)
        return np.delete(f_cons+f_dis,i_kappa2)
    
    def planar_jacobian_with_dissipation(self,X,tau_e,tau_a,p):
        shape = 2*(self.N_planar+self.M) + 1
        msg = "Input X has dimensions {} but should be size ({},)".format(X.shape,shape)
        assert X.size==2*(self.N_planar+self.M) + 1, msg
        i_kappa2 = self.N_planar
        X_full = np.insert(X,i_kappa2,0.)
        jac_cons = self.planar_full_jacobian(X_full)
        jac_dis = self.planar_dissipation_jacobian(X_full,tau_e,tau_a,p)
        return np.delete(np.delete(jac_cons+jac_dis,i_kappa2,axis=0),i_kappa2,axis=1)    
    
    def planar_flow_and_jacobian_with_dissipation(self,X,tau_e,tau_a,p):
        shape = 2*(self.N_planar+self.M) + 1
        msg = "Input X has dimensions {} but should be size ({},)".format(X.shape,shape)
        assert X.size==2*(self.N_planar+self.M) + 1, msg
        i_kappa2 = self.N_planar
        X_full = np.insert(X,i_kappa2,0.)
        f_cons,jac_cons = self.planar_full_flow_and_jacobian(X_full)
        
        f_dis = self.planar_dissipation_flow(X_full,tau_e,tau_a,p)
        jac_dis = self.planar_dissipation_jacobian(X_full,tau_e,tau_a,p)
        
        f_tot = np.delete(f_cons+f_dis,i_kappa2)
        jac_tot = np.delete(np.delete(jac_cons+jac_dis,i_kappa2,axis=0),i_kappa2,axis=1)
        return f_tot,jac_tot
    
    def planar_full_flow(self,X):
        return _real_flow(X,self.planar_full_complex_flow_list,self.N_planar,self.M+1)
    
    def planar_full_jacobian(self,X):
        return _real_jacobian(X,self.planar_full_jacobian_list,self.N_planar,self.M+1)

    def planar_full_flow_and_jacobian(self,X):
        Ndim = 2*(self.N_planar + self.M+1)
        assert X.size==Ndim, "Input 'X' has dimension ({},) but should be dimension ({},)".format(X.size,Ndim)
        return _real_flow_and_jacobian(X,
                                       self.planar_full_complex_flow_list,
                                       self.planar_full_jacobian_list,
                                       self.N_planar,
                                       self.M+1
                                       )
    def hamiltonian(self,X):
        return _real_hamiltonian(X,self.h_series,self.N,self.M)    
    
    def flow(self,X):
        return _real_flow(X,self.complex_flow_list,self.N,self.M)
    
    def jacobian(self,X):
        return _real_jacobian(X,self.jacobian_list,self.N,self.M)
    
    def flow_and_jacobian(self,X):
        return _real_flow_and_jacobian(X,self.complex_flow_list,self.jacobian_list,self.N,self.M)

    def planar_hamiltonian(self,X):
        Ndim = 2*(self.N_planar + self.M)
        assert X.size==Ndim, "Input 'X' has dimension ({},) but should be dimension ({},)".format(X.size,Ndim)
        return _real_hamiltonian(X,self.h_planar_series,self.N_planar,self.M)    
        
    def planar_flow(self,X):
        Ndim = 2*(self.N_planar + self.M)
        assert X.size==Ndim, "Input 'X' has dimension ({},) but should be dimension ({},)".format(X.size,Ndim)
        return _real_flow(X,self.planar_complex_flow_list,self.N_planar,self.M)
    
    def planar_jacobian(self,X):
        Ndim = 2*(self.N_planar + self.M)
        assert X.size==Ndim, "Input 'X' has dimension ({},) but should be dimension ({},)".format(X.size,Ndim)
        return _real_jacobian(X,self.planar_jacobian_list,self.N_planar,self.M)
    
    def planar_flow_and_jacobian(self,X):
        Ndim = 2*(self.N_planar + self.M)
        assert X.size==Ndim, "Input 'X' has dimension ({},) but should be dimension ({},)".format(X.size,Ndim)
        return _real_flow_and_jacobian(X,self.planar_complex_flow_list,self.planar_jacobian_list,self.N//2,self.M)
    
    def pvars_to_real_vars(self,pvars,planar = True):
        ps = pvars.particles[1:]
        actions = self.Tmtrx_inv.T @ ([p.Lambda for p in ps] + [p.Gamma for p in ps] + [p.Q for p in ps])
        angles = self.Tmtrx @ ([p.l for p in ps] + [p.gamma for p in ps] + [p.q for p in ps])
        K1 = actions[0]
        rho = self.Lambda0s / self.Lambda0s[0]
        nu_dot_rho = self.Tmtrx_inv.T[0,:self.N_planar] @ rho
        L10 = K1 / nu_dot_rho
        ref_actions = np.zeros(actions.size)
        ref_actions[:rho.size] = rho
        ref_actions =  self.Tmtrx_inv.T @ ref_actions
        actions = (actions / L10) 
        dK2 = (actions-ref_actions)[1]

        P = (actions-ref_actions)[2:2+self.M]
        Q = angles[2:2+self.M]
        z= np.sqrt(2*actions[2+self.M:])*np.exp(1j*angles[2+self.M:])
        x = np.real(z)
        y = np.imag(z)
        if planar:
            return np.concatenate((y[:self.N_planar],Q,x[:self.N_planar],P)), dK2
        else:
            return np.concatenate((y,Q,x,P)), dK2




    
        
    def real_vars_to_pvars(self,real_vars,kappa1=0,kappa2=0,dK2=None):
        Npl = self.pham.N - 1
        actions,angles = np.zeros((2,self.N_full+self.M_full))
        y = real_vars[:self.N]
        x = real_vars[self.N+self.M:2*self.N+self.M]
        P = real_vars[2*self.N+self.M:]
        Q = real_vars[self.N:self.N+self.M]
        angles[0] = kappa1
        angles[1] = kappa2
        angles[2:2+self.M] = Q
        angles[2+self.M:] = np.arctan2(y,x)

        # Tinv_transpose = self.Tmtrx_inv.T
        # rho = self.Lambda0s/self.Lambda0s[0]
        # nu_vec = Tinv_transpose[0,:Npl]
        # actions[0] = nu_vec @ rho
        actions[1] = dK2 if dK2 else self.dK2
        actions[2:2+self.M] = P
        actions[2+self.M:] = 0.5 * (x**2 + y**2)
        actions *= self.action_scale
        
        
        old_actions = self.Tmtrx.T @ actions
        old_angles = self.Tmtrx_inv @ angles
        Lambda0s = np.array([self.pham.H_params[L0] for L0 in self.pham.Lambda0s[1:]])
        Lambdas = Lambda0s + old_actions[:Npl]
        Gammas = old_actions[Npl:2*Npl]
        Qs = old_actions[2*Npl:]
        lambdas,gammas,qs = old_angles.reshape(3,Npl)
        kappas = np.sqrt(2*Gammas) * np.cos(gammas)
        etas = np.sqrt(2*Gammas) * np.sin(gammas)
        sigmas = np.sqrt(2*Qs) * np.cos(qs)
        rhos = np.sqrt(2*Qs) * np.sin(qs)
        pvars = self.pham.state.copy()
        vals = np.array([(l,h,r) for l,h,r in zip(lambdas,etas,rhos)] + [(L,k,s) for L,k,s in zip(Lambdas,kappas,sigmas)]).reshape(-1)
        pvars.values = vals
        return pvars
    
    def real_planar_vars_to_pvars(self,real_vars,kappa1=0,kappa2=0,**kwargs):
        zero_N_planar = np.zeros(self.N_planar)
        y = np.concatenate((real_vars[:self.N_planar],zero_N_planar))
        x = np.concatenate((real_vars[self.N_planar+self.M:2*self.N_planar+self.M],zero_N_planar))
        P = real_vars[2*self.N_planar+self.M:]
        Q = real_vars[self.N_planar:self.N_planar+self.M]
        real_vars = np.concatenate((y,Q,x,P))
        return self.real_vars_to_pvars(real_vars,kappa1=kappa1,kappa2=kappa2,**kwargs)

    def Poincare_to_real_vars(self,pvars):
        actions = [p.Lambda-L0 for p,L0 in zip(pvars.particles[1:],self.Lambda0s)] + [p.Gamma for p in pvars.particles[1:]] + [p.Q for p in pvars.particles[1:]]
        angles = [p.l for p in pvars.particles[1:]] + [p.gamma for p in pvars.particles[1:]] + [p.q for p in pvars.particles[1:]]
        new_angles = self.Tmtrx @ angles
        new_actions = self.Tmtrx_inv.T @ actions
        new_actions *= 1/self.action_scale
        x = np.sqrt(2*new_actions[2+self.M:]) * np.cos(new_angles[2+self.M:])
        y = np.sqrt(2*new_actions[2+self.M:]) * np.sin(new_angles[2+self.M:])
        Q = np.mod(new_angles[2:2+self.M],2*np.pi)
        P = new_actions[2:2+self.M]
        return np.concatenate((y,Q,x,P))

    def Poincare_to_real_planar_vars(self,pvars):
        real_vars = self.Poincare_to_real_vars(pvars)
        return self.real_vars_to_real_planar_vars(real_vars)

    def real_vars_to_real_planar_vars(self,real_vars):
        y = real_vars[:self.N]
        x = real_vars[self.N+self.M:2*self.N+self.M]
        P = real_vars[2*self.N+self.M:]
        Q = real_vars[self.N:self.N+self.M]
        return np.concatenate((y[:self.N_planar],Q,x[:self.N_planar],P))

def get_chain_hpert(resonances, masses, max_order,max_order_dl = 1):
    """
    Get Poisson series representation of interaction hamiltonian for a resonant chain. 
    Terms up to a user-specified maximum order in eccentricities and inclinations are included.

    Parameters
    ----------
    resonances : list
        List of integer tuples [(j1,k1),...,(jN,kN)] specifying resonances between adjacent chain members.
    masses : ndarray
        planet masses
    max_order : int
        maximum order of eccentricity/inclination expansion
    max_order_dl : int, optional
        Maximum order of eccentricity/inclination expansion terms for which terms linear in delta-Lambda are included.
        By default 1

    Returns
    -------
    celmech.poisson_series.PoissonSeries
        Series representation of Hamiltonian
    """
    sim = get_chain_rebound_sim(resonances,masses)
    pvars = cm.Poincare.from_Simulation(sim)
    pham = cm.PoincareHamiltonian(pvars)
    periods = []
    period = 1
    periods.append(period)
    for j,k in resonances:
        period *= j/sp.S(j-k)
        periods.append(period)
    hpert_series_terms = []

    ei_order = lambda k,nu: np.sum(np.abs(k[2:])) + 2*np.sum(nu)
    for i1,p1 in enumerate(periods):
        for i2,p2 in zip(range(i1+1,len(periods)),periods[i1+1:]):
            pratio = p2/p1
            p,q = int(sp.numer(pratio)),int(sp.numer(pratio) - sp.denom(pratio))
            resonant_terms = cm.disturbing_function.list_resonance_terms(p,q,max_order = max_order)
            if len(resonant_terms)>0:
                print("adding terms for {}:{} MMR between {} and {}".format(p,p-q,i1+1,i2+1))
            for k,nu in resonant_terms:
                hpert_series_terms += cm.poisson_series.DFTerm_as_PSterms(pham,i1+1,i2+1,k,nu,(0,0))
                if ei_order(k,nu) <= max_order_dl:
                    hpert_series_terms += cm.poisson_series.DFTerm_as_PSterms(pham,i1+1,i2+1,k,nu,(1,0))
                    hpert_series_terms += cm.poisson_series.DFTerm_as_PSterms(pham,i1+1,i2+1,k,nu,(0,1))

    secular_terms = cm.disturbing_function.list_secular_terms(min_order=0,max_order = max_order)
    for k,nu in secular_terms:
        for i1 in range(1,sim.N):
            for i2 in range(i1+1,sim.N):
                hpert_series_terms += cm.poisson_series.DFTerm_as_PSterms(pham,i1,i2,k,nu,(0,0))
                if ei_order(k,nu) <= max_order_dl:
                    hpert_series_terms += cm.poisson_series.DFTerm_as_PSterms(pham,i1,i2,k,nu,(1,0))
                    hpert_series_terms += cm.poisson_series.DFTerm_as_PSterms(pham,i1,i2,k,nu,(0,1))
    return PoissonSeries.from_PSTerms(hpert_series_terms)
