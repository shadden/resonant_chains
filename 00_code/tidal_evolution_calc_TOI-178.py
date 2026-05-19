import numpy as np
from resonant_chain_utils import damping_rate_tides
from pathlib import Path
from resonant_chains import get_chain_hpert, ResonantChainPoissonSeries
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
system_name = "TOI-178"
cache_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") / system_name
max_order = 3
fname = f"{system_name}_maxorder{max_order}.npz"
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

gamma_e_values = damping_rate_tides(masses,resonances,radii,Qs=100)
tau_e_values = 1/gamma_e_values * Pinner_in_yr
print("Tidal damping timescales (in Myr):", tau_e_values/1e6)
tau_a_values = np.inf * np.ones(len(resonances)+1)
##############################################################

hpert = get_chain_hpert(data["resonances"],data["masses"],data["max_order"],1)
rc = ResonantChainPoissonSeries(data["resonances"],data["masses"],hpert,max_order = data["max_order"])
ell_eqs = np.load(cache_dir / "elliptic_eq_data.npz")
diss_data = np.load(cache_dir/"TOI-178_dissipation_info.npz")
eqs_extended = np.insert(ell_eqs['eqs'],2 * rc.N_planar + rc.M,ell_eqs['dK2vals'],axis = 1)

dK2dt_tide = np.array([
    rc.planar_flow_with_dissipation(x,tau_e_values,tau_a_values,1)[2 * rc.N_planar + rc.M] 
    for x in eqs_extended
])



f = interp1d(ell_eqs['dK2vals'],dK2dt_tide,kind='cubic',fill_value='extrapolate')
Delta_of_K2 = interp1d(ell_eqs['dK2vals'],ell_eqs['Periods'].T[1]/ell_eqs['Periods'].T[0]/2-1,kind='cubic',fill_value='extrapolate')
K2_final= ell_eqs['dK2vals'][0]
K2_initials = np.linspace(0, K2_final, 20)
T_evolve = np.array([quad(lambda K2: 1/f(K2),K2_initial,K2_final)[0] for K2_initial in K2_initials])
fig,ax = plt.subplots(2,1,figsize=(8,6),sharex=True)

for i in [0,1,2]:
    si_of_K2 = interp1d(ell_eqs['dK2vals'],(diss_data['damping_rate_coeffs'] @ (1/tau_e_values))[:,i])
    dlog_Ai = 0.5 * np.array([quad(lambda K2: 2*si_of_K2(K2)/f(K2),K0,K2_final)[0] for K0 in K2_initials])
    l,=ax[1].plot(
        Delta_of_K2(K2_initials),
        measured_amps[i]/np.exp(dlog_Ai),lw=3,label = f"Mode {i+1}")
    ax[1].axhline(measured_amps[i],ls='--',color=l.get_color())
ax[1].set_ylim(0,180)
ax[1].set_ylabel(r"$\delta\phi_{1}$ [$^{\circ}$]",fontsize=16)
ax[1].set_xlabel(r'Initial $\Delta_{1,2}$')

ax[1].legend(loc='upper left',fontsize=12)

ax[0].plot(Delta_of_K2(K2_initials),T_evolve/1e9,color='k',lw=3)
ax[0].set_title(r'TOI-178 tidal evolution')

ax[0].set_ylabel(r"Time to evolve to equilibrium [Gyr$\times \left(\frac{Q'}{100}\right)$]")
plt.show()