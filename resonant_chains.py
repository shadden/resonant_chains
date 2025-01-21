
import celmech as cm
import numpy as np
from celmech.canonical_transformations import CanonicalTransformation
import sympy as sp
import rebound as rb
from celmech.nbody_simulation_utilities import add_canonical_heliocentric_elements_particle
import warnings
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

def _swap_entries(list,i1,i2):
    list[i1],list[i2] = list[i2],list[i1]

class ResonantChain():
    def __init__(self,resonances, masses,add_MMR_terms_kwargs = {}, add_secular_terms_kwargs = {}):
        self.N = len(masses) + 1
        self.resonances = resonances
        self.masses = np.array(masses)
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
        #self.Npars.pop(kappa1)
        #self.Npars.pop(kappa2)
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

    def _initialize_ct(self,ct):
        canonical_vars_matrix = sp.Matrix([
            sp.symbols("lambda{0},eta{0},rho{0},Lambda{0},kappa{0},sigma{0}".format(i)) for i in range(1,self.N)
        ])
        canonical_vars_matrix = ct.old_to_new(canonical_vars_matrix)
        canonical_vars_matrix = _explicit_mass_dependence_substitution(canonical_vars_matrix,self.N)
        canonical_vars_matrix = canonical_vars_matrix.xreplace(self.Npars)
        self._canonical_vars_matrix_func = sp.lambdify(self.arg_symbols,canonical_vars_matrix)
    def kappa1_dot(self,x,dK2val,masses):
        return self._kappa1_dot_func(*x,dK2val,*masses)
    def kappa2_dot(self,x,dK2val,masses):
        return self._kappa2_dot_func(*x,dK2val,*masses)
    def f(self,x,dK2val,masses):
        return self._flow_func(*x,dK2val,*masses).reshape(-1)
    def Df(self,x,dK2val,masses):
        return self._jac_func(*x,dK2val,*masses)
    def solve_eq_x1(self,x1,masses,guess = None):
        if guess is None:
            guess = np.zeros(self.ham.N_dim)
        f = lambda x: self._flow_func_x1_par(*x,x1,*masses).reshape(-1)
        Df = lambda x: self._jac_func_x1_par(*x,x1,*masses)
        soln = newton_solve(f,Df,guess)
        dK2 = soln[self.i_x1]
        soln[self.i_x1] = x1
        return soln, dK2, masses
    def get_Poincare(self,x,dK2val,masses):
        cvars_matrix = self._canonical_vars_matrix_func(*x,dK2val,*masses)
        particles = []
        varnames = "l,eta,rho,Lambda,kappa,sigma".split(",")
        for m,cvars in zip(masses,cvars_matrix): 
            kwargs = dict(zip(varnames,cvars))
            particle = cm.poincare.PoincareParticle(G=self.G,m=m,Mstar=self.Mstar,**kwargs)
            particles.append(particle)
        pvars = cm.Poincare(self.G,particles)
        return pvars

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

def resonant_chain_variables_transformation_matrix(resonances):
    # number of planets
    N = len(resonances) + 1
    A = np.zeros((3*N,3*N),dtype=int)
    plast,qlast = resonances[-1]
    A[0,N-2:N] = 1,-1 #-1,1
    A[1,N-2] = qlast - plast
    A[1,N-1] = plast
    for i in range(N-2):
        resvec1,resvec2 = np.zeros((2,N),dtype=int)
        p1,q1 = resonances[i]
        p2,q2 = resonances[i+1]
        resvec1[i] = q1 - p1
        resvec1[i+1] = p1
        resvec2[i+1] = q2 - p2
        resvec2[i+2] = p2
        row = resvec2 - resvec1
        A[i+2,:N] = row #// np.gcd.reduce(row)
    A[N:,N-2] = qlast - plast
    A[N:,N-1] = plast
    A[N:,N:] = np.eye(2*N,dtype=int)
    return A
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
    for i,pq in enumerate(resonances):
        pham.add_MMR_terms(*pq,indexIn=i+1,indexOut=i+2,**add_MMR_terms_kwargs)
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