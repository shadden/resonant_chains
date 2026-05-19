from resonantstate.data_download  import get_metadata_observations, download_observations_samples
from resonantstate.analyse_samples import *
from resonant_chain_utils import Deltas_to_pvars
from resonant_chains import get_chain_hpert, ResonantChainPoissonSeries,newton_solve2
# load pandas data on system
df_all_obs = get_metadata_observations()
system_name = "TRAPPIST-1"

import pandas as pd
import requests
from io import StringIO

from pathlib import Path
import numpy as np
def get_planet_params():
    # ADQL query against the NASA Exoplanet Archive TAP service.
    # We use the PS table because it stores one row per planet per reference.
    query_default = r"""
    SELECT
        pl_name,
        pl_letter,
        hostname,
        pl_orbper,
        pl_orbpererr1,
        pl_orbpererr2,
        pl_bmasse,
        pl_bmasseerr1,
        pl_bmasseerr2,
        st_mass,
        st_masserr1,
        st_masserr2,
        pl_refname
    FROM ps
    WHERE hostname = 'TRAPPIST-1'
        AND default_flag = 1
    ORDER BY pl_orbper
    """

    tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    response = requests.get(
        tap_url,
        params={
            "query": query_default,
            "format": "csv",
        },
        timeout=60,
    )
    df = pd.read_csv(StringIO(response.text))
    masses = (df.pl_bmasse / df.st_mass) * 3e-6
    periods = df.pl_orbper.values
    order = np.argsort(periods)
    return masses[order],periods[order]

    
# ----------------------------
# choose a cache/save filename
# ----------------------------
cache_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") / system_name
cache_dir.mkdir(parents=True, exist_ok=True)
max_order = 3

# make the filename depend on what matters
fname = f"{system_name}_maxorder{max_order}.npz"
save_path = cache_dir / fname

# ----------------------------
# compute-or-load
# ----------------------------
if not save_path.exists():
    
    masses,periods = get_planet_params()
    # Primoridal resonance suggested by Pichierri+24 
    # 3:2 – 3:2 – 3:2 – 3:2 – 4:3 – 3:2
    resonances = [(3,1)] * 4 + [(4,1),(3,1)]
    

    # --- compute ---

    hpert = get_chain_hpert(resonances,masses,max_order,1)
    rc = ResonantChainPoissonSeries(resonances,masses,hpert,max_order = max_order)
    pvars = Deltas_to_pvars([0.001 for _ in resonances],resonances,masses)
    rvars, dK20 = rc.pvars_to_real_vars(pvars)
    print(dK20)
    # Guess initial three-body angles close to 180deg for angle in seigel-fabrycky form
    for i,(rin,rout) in enumerate(zip(resonances,resonances[1:])):
        jin,kin = rin
        jout,kout = rout
        k_SF = np.array([0,kout-jout,jout]) - np.array([kin-jin,jin,0]) # seigel-fabrycky k vector 
        k_canon = rc.Tmtrx[2+i,i:i+3]  # canonical variable k vector
        ratio = k_SF / k_canon
        assert np.all(np.isclose(ratio,ratio[0]))
        ratio = ratio[0]
        rvars[rc.N_planar + i] = 0.5 * np.pi / ratio
    print("initial guess:",rvars)
    rc.dK2 = dK20
    eqC = newton_solve2(rc.planar_flow_and_jacobian, rvars)
    f, Df = rc.planar_flow_and_jacobian(eqC)
    eigs = np.linalg.eigvals(Df)
    assert np.all(np.isclose(np.real(eigs), 0)), "Non-imaginary eigenvalues found."
    
    # --- save ---
    # .npz stores arrays; wrap scalars with np.asarray to be safe
    np.savez(
        save_path,
        dK2=np.asarray(dK20),
        equilibrium=np.asarray(eqC),
        resonances=np.asarray(resonances, dtype=int),
        masses=np.asarray(masses),
        periods=np.asarray(periods),
        system_name=np.asarray(system_name),
        max_order=np.asarray(max_order),
    )

    print(f"Saved cached equilibrium to: {save_path}")

else:
    # --- load ---
    data = np.load(save_path, allow_pickle=True)
    dK20 = data["dK2"]
    eqC = data["equilibrium"]
    resonances = data['resonances']
    hpert = get_chain_hpert(data["resonances"],data["masses"],data["max_order"],1)
    rc = ResonantChainPoissonSeries(data["resonances"],data["masses"],hpert,max_order = data["max_order"])
    rc.dK2 = dK20
    print(f"Loaded cached equilibrium from: {save_path}")

    f, Df = rc.planar_flow_and_jacobian(eqC)
    eigs = np.linalg.eigvals(Df)
    assert np.all(np.isclose(np.real(eigs), 0)), "Non-imaginary eigenvalues found."

# payload now exists in both branches
print("dK2 =", dK20)
print("equilibrium shape =", np.asarray(eqC).shape)

from celmech.disturbing_function import get_fg_coefficients

guess = eqC.copy()
dK2vals = dK20 + np.linspace(0.0075,-0.0075,80) 
eqsC = np.zeros((dK2vals.size,guess.size))
jacobians = np.zeros((dK2vals.size,guess.size,guess.size))
freqs = np.zeros((dK2vals.size,guess.size//2))
lmbdas = np.zeros((dK2vals.size,guess.size//2))
eccs = np.zeros((dK2vals.size,rc.N_planar))
Periods = np.zeros((dK2vals.size,rc.N_planar))
kappa2_dot = np.zeros(dK2vals.size)
Z = np.zeros((dK2vals.size,rc.N_planar-1))
f_res,g_res = np.transpose([get_fg_coefficients(j,k) for j,k in resonances])

for i,dK2val in enumerate(dK2vals):
    rc.dK2 = dK2val
    guess = newton_solve2(rc.planar_flow_and_jacobian,guess)
    eqsC[i] = guess
    jac = rc.planar_jacobian(guess)
    jacobians[i] = jac
    eigs = np.linalg.eigvals(jac)
    freq = np.sort(np.imag(eigs))
    freqs[i] = freq[freq.size//2:]
    lmbda = np.sort(np.real(eigs))
    lmbdas[i] = lmbda[freq.size//2:]
    pvars = rc.real_planar_vars_to_pvars(guess)
    eccs[i] = [p.e for p in pvars.particles[1:]]
    Periods[i] = [p.P for p in pvars.particles[1:]]
    X=np.insert(guess,[rc.N_planar,2*rc.N_planar+rc.M],[0,rc.dK2])
    kappa2_dot[i] = rc.planar_full_flow(X)[rc.N_planar]
    for l,jk in enumerate(resonances):
        pin = pvars.particles[1+l]
        pout = pvars.particles[2+l]
        zin = pin.e * np.exp(1j * pin.pomega)
        zout = pout.e * np.exp(1j * pout.pomega)
        Z[i,l] = np.abs((f_res[l] * zin + g_res[l] * zout) / np.sqrt(f_res[l] ** 2 + g_res[l] ** 2))

np.savez(
    cache_dir / "elliptic_eq_data.npz",
    dK2vals = dK2vals,
    eqs = eqsC,
    Periods = Periods,
    freqs = freqs,
    jacobians = jacobians,
    kappa2_dot = kappa2_dot,
    Z = Z,
    eccs = eccs
)

