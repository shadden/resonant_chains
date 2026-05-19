from resonant_chains import get_chain_hpert, ResonantChainPoissonSeries, newton_solve2
import reboundx as rbx
from pathlib import Path
import numpy as np
import rebound as rb
system_name = "Kepler-60"

try:
    sim = rb.Simulation("kepler-60-long-damping.sa")
    print("found simulation!")
except:
    # ----------------------------
    # choose a cache/save filename
    # ----------------------------
    cache_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") / system_name
    max_order = 3

    # make the filename depend on what matters
    fname = f"{system_name}_maxorder{max_order}.npz"
    save_path = cache_dir / fname

    data = np.load(save_path, allow_pickle=True)
    dK20 = data["dK2"]
    eqC = data["equilibrium"]
    resonances = data['resonances']
    hpert = get_chain_hpert(data["resonances"],data["masses"],max_order,1)
    rc = ResonantChainPoissonSeries(
        data["resonances"],
        data["masses"],
        hpert,
        max_order = max_order
    )

    eq_data = np.load(cache_dir / "elliptic_eq_data.npz")

    # ----------------------------
    # choose initial equilibrium 
    # ----------------------------

    i0 = np.argmin(np.abs(eq_data['dK2vals'])) + 10
    rc.dK2 = eq_data['dK2vals'][i0]
    pvars0 = rc.real_planar_vars_to_pvars(eq_data['eqs'][i0])

    # ----------------------------
    # initialize rebound simulation
    # ----------------------------

    sim = pvars0.to_Simulation()
    sim.integrator='whfast'
    sim.dt = sim.particles[1].P / 25.
    sim.t=0

# ----------------------------
# add eccentricity damping
# ----------------------------

extras = rbx.Extras(sim)
mod = extras.load_operator("modify_orbits_direct")
extras.add_operator(mod)
for p in sim.particles[1:]:
    p.params['tau_e'] = -3e4 * 2 * np.pi


    # ----------------------------
    # run simulation
    # ----------------------------
print(f"Starting simulation from t={sim.t}")
Tfin = 2 * np.pi * 3e7
sim.save_to_file("kepler-60-long-damping.sa",interval = Tfin / 5500)
sim.integrate(2*Tfin)