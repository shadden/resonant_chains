import numpy as np
from resonant_chain_utils import damping_rate_tides
from pathlib import Path
from resonant_chains import get_chain_hpert, ResonantChainPoissonSeries
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
system_name = "K2-138"
cache_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") / system_name
max_order = 3
fname = f"{system_name}_maxorder{max_order}.npz"
save_path = cache_dir / fname


def select_by_nearest_key(d,val):
    keys_arr = np.array(list(d.keys()))
    nearest_key = keys_arr[np.argmin(np.abs(keys_arr - val))]
    return nearest_key,d[nearest_key]
############################################################
data = np.load(save_path, allow_pickle=True)
dK20 = data["dK2"]
eqC = data["equilibrium"]
resonances = data['resonances']
masses = data['masses']
radii = np.array([1.51,2.3,2.39,3.39,3.01])
REARTH_IN_AU = 0.0000426352
a_inner = 0.03385
radii *= REARTH_IN_AU / a_inner
Pinner_in_yr = 2.35 / 365.25

gamma_e_values = damping_rate_tides(masses,resonances,radii,Qs=100)
tau_e_values = 1/gamma_e_values * Pinner_in_yr
print("Tidal damping timescales (in Myr):", tau_e_values/1e6)
tau_a_values = np.inf * np.ones(len(resonances)+1)
##############################################################

hpert = get_chain_hpert(data["resonances"],data["masses"],data["max_order"],1)
rc = ResonantChainPoissonSeries(data["resonances"],data["masses"],hpert,max_order = data["max_order"])
ell_eqs = np.load(cache_dir / "elliptic_eq_data.npz")
diss_data = np.load(cache_dir/"K2-138_dissipation_info.npz")
eqs_extended = np.insert(ell_eqs['eqs'],2 * rc.N_planar + rc.M,ell_eqs['dK2vals'],axis = 1)

dK2dt_tide = np.array([
    rc.planar_flow_with_dissipation(x,tau_e_values,tau_a_values,1)[2 * rc.N_planar + rc.M] 
    for x in eqs_extended
])


j1,k1 = data["resonances"][0]
f = interp1d(ell_eqs['dK2vals'],dK2dt_tide,kind='cubic',fill_value='extrapolate')
Delta_of_K2 = interp1d(ell_eqs['dK2vals'],(j1-k1)*ell_eqs['Periods'].T[1]/ell_eqs['Periods'].T[0]/j1-1,kind='cubic',fill_value='extrapolate')
K2_final= ell_eqs['dK2vals'][0]
K2_initials = np.linspace(0, K2_final, 20)
T_evolve = np.array([quad(lambda K2: 1/f(K2),K2_initial,K2_final)[0] for K2_initial in K2_initials])
fig,ax = plt.subplots(1,1,figsize=(8,6),sharex=True)

ax.plot(Delta_of_K2(K2_initials),T_evolve/1e9,color='k',lw=3)
ax.set_title(r'K2-138 tidal evolution')
ax.set_ylabel(r"Time to evolve to equilibrium [Gyr$\times \left(\frac{Q'}{100}\right)$]")
plt.show()