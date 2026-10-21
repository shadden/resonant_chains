import numpy as np
from resonant_chain_utils import damping_rate_tides
from pathlib import Path
from resonant_chains import get_chain_hpert, ResonantChainPoissonSeries
from resonant_chain_utils import relative_damping_rate_tau_wave
from scipy.optimize import root_scalar
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
system_name = "TOI-178"
cache_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") / system_name
max_order = 3
h0_order = 4
#TOI-178_maxorder3_h0order2.npz
fname = f"{system_name}_maxorder{max_order}_h0order{h0_order}.npz"
save_path = cache_dir / fname


def select_by_nearest_key(d,val):
    keys_arr = np.array(list(d.keys()))
    nearest_key = keys_arr[np.argmin(np.abs(keys_arr - val))]
    return nearest_key,d[nearest_key]
############################################################
# fmft data
measured_amps = np.array([ 8.72968564, 14.68497443, 11.47164177])    
############################################################
data = np.load(save_path, allow_pickle=True)
dK20 = data["dK2"]
eqC = data["equilibrium"]
resonances = data['resonances']
masses = data['masses']
radii = np.array([1.669,2.572,2.207,2.287,2.87])
REARTH_IN_AU = 0.0000426352
a_inner = 0.0370
radii *= REARTH_IN_AU / a_inner
Pinner_in_yr = 3.238450 / 365.25

gamma_e_values = damping_rate_tides(masses,resonances,radii,Qs=1e2)
tau_e_values = 1/gamma_e_values * Pinner_in_yr
print("Tidal damping timescales (in Gyr):", tau_e_values/1e9)
tau_a_values = np.inf * np.ones(len(resonances)+1)
##############################################################

hpert = get_chain_hpert(data["resonances"],data["masses"],data["max_order"],1)
rc = ResonantChainPoissonSeries(data["resonances"],data["masses"],hpert,max_order = data["max_order"])
ell_eqs = np.load(cache_dir / f"elliptic_eq_data_maxorder{max_order}_h0order{h0_order}.npz")
diss_data = np.load(cache_dir / f"TOI-178_dissipation_info_maxorder{max_order}_h0order{h0_order}.npz")
eqs_extended = np.insert(ell_eqs['eqs'],2 * rc.N_planar + rc.M,ell_eqs['dK2vals'],axis = 1)

dK2dt_tide = np.array([
    rc.planar_flow_with_dissipation(x,tau_e_values,tau_a_values,1)[2 * rc.N_planar + rc.M] 
    for x in eqs_extended
])

f = interp1d(ell_eqs['dK2vals'],dK2dt_tide,kind='cubic',fill_value='extrapolate')
Delta_of_K2 = interp1d(ell_eqs['dK2vals'],ell_eqs['Periods'].T[1]/ell_eqs['Periods'].T[0]/2-1,kind='cubic',fill_value='extrapolate')
K2_of_Delta = interp1d(ell_eqs['Periods'].T[1]/ell_eqs['Periods'].T[0]/2-1,ell_eqs['dK2vals'],kind='cubic',fill_value='extrapolate')
Delta_obs=0.0126
K2_final= K2_of_Delta(Delta_obs)
K2_initials = np.linspace(0, K2_final, 20)
T_evolve = np.array([quad(lambda K2: 1/f(K2),K2_initial,K2_final)[0] for K2_initial in K2_initials])

# compute disk scale height as function of initial Delta
# set up relative damping timescales for disk migration
gamma_rel_a,gamma_rel_e = relative_damping_rate_tau_wave(masses,resonances)
tau_a_rel = 1e4/gamma_rel_a
tau_e_rel = 1e4/gamma_rel_e
tau_a_rel[0] = np.inf

h_solns = []
guess = 0.05
for eq_ext in eqs_extended:
    alpha = 1.0
    K_of_h = lambda h: 0.78 / (2.7 + 1.1*alpha) / h / h
    g = lambda h: rc.planar_flow_with_dissipation(
        eq_ext,
        tau_e_rel,
        (K_of_h(h))*tau_a_rel,
        1
    )[2 * rc.N_planar + rc.M]
    ## assert g(5), "Negative derivative at log10K=5; expand search range."
    rt = root_scalar(g,bracket = [0.001,0.5],x0 = guess)
    guess = rt.root
    h_solns.append(guess)

h_solns = np.array(h_solns)
msk = ell_eqs['sundman_convergence']


# make plots
fig,ax = plt.subplots(2,1,figsize=(8,6),sharex=True)
ax_h = ax[0].twinx()
ax_h.plot(
    Delta_of_K2(ell_eqs["dK2vals"][msk]),
    h_solns[msk],
    color="tab:blue",
    lw=3
)

ax_h.set_yscale("log")
ax_h.set_ylabel(
    r"Disk scale height, $h(a_1)$",
    color="tab:blue",
    fontsize=14
)

for i in [0,1,2]:
    si_of_K2 = interp1d(ell_eqs['dK2vals'],(diss_data['damping_rate_coeffs'] @ (1/tau_e_values))[:,i])
    dlog_Ai = 0.5 * np.array([quad(lambda K2: 2*si_of_K2(K2)/f(K2),K0,K2_final)[0] for K0 in K2_initials])
    l,=ax[1].plot(
        Delta_of_K2(K2_initials),
        measured_amps[i]/np.exp(dlog_Ai),lw=3,label = f"Mode {i+1}"
        )
    Delta_at_180 = interp1d(measured_amps[i]/np.exp(dlog_Ai),Delta_of_K2(K2_initials),kind='cubic',fill_value='extrapolate')(180)
    print(f"Initial Delta for mode {i+1} to reach 180 deg:",Delta_at_180)
    ax[1].axhline(measured_amps[i],ls='--',color=l.get_color())
ax[1].set_ylim(-2,182)
ax[1].set_yticks([0,45,90,135,180])
ax[1].set_ylabel(r"$\delta\phi_{1}$ [$^{\circ}$]",fontsize=16)

ax[1].legend(loc='upper left',fontsize=12)

ax[0].plot(Delta_of_K2(K2_initials),T_evolve/1e9,color='k',lw=3)

initial_Delta = interp1d(T_evolve/1e9,Delta_of_K2(K2_initials),kind='cubic',fill_value='extrapolate')(6)
initial_K2 = interp1d(T_evolve/1e9,K2_initials,kind='cubic',fill_value='extrapolate')(6)
print("Initial Delta  for 6 Gyr evolution:",initial_Delta)
print("Initial K2  for 6 Gyr evolution:",initial_K2)
ax[0].set_title(r'TOI-178 Tidal Evolution',fontsize=16)

ax[0].set_ylabel(r"Evolution time [Gyr$\times \left(\frac{Q'}{100}\right)$]",fontsize=16)
for a in ax:
    plt.sca(a)
    plt.tick_params(axis='both', which='major', labelsize=14,direction='in',length=6,width=1.5)
# plt.show()
ax[1].set_xlabel(r"Initial $\Delta_{{1,2}}$",fontsize=14)
ax[0].set_ylabel(r"$t_{\rm evolve} \times \left(\frac{Q'}{100}\right)$ [Gyr]",fontsize=14)

# Tick formatting
for a in ax:
    a.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=8,
        labelsize=12
    )
    a.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=6
    )
ax_h.tick_params(
    axis="y",
    which="major",
    direction="in",
    length=8,
    labelsize=12,
    colors="tab:blue"
)
ax_h.tick_params(
    axis="y",
    which="minor",
    direction="in",
    length=6,
    colors="tab:blue"
)
ax_h.spines["right"].set_color("tab:blue")
plt.tight_layout()


plt.savefig("/Users/hadden/Papers/10_chain_dynamics/04_figures/TOI-178_tidal_evolution.png",bbox_inches='tight',dpi=300)