import rebound as rb
import numpy as np
from celmech.nbody_simulation_utilities import add_canonical_heliocentric_elements_particle
import rebound as rb
import celmech as cm

def Deltas_to_pvars(Deltas,resonances,masses):
    sim = rb.Simulation()
    sim.add(m=1)
    Period = 1
    add_canonical_heliocentric_elements_particle(masses[0],{'a' : Period**(2/3)},sim)
    for Delta,mass,pq in zip(Deltas,masses[1:],resonances):
        p,q = pq
        Period *= (1+Delta)*p/(p-q)
        add_canonical_heliocentric_elements_particle(mass,{'a' : Period**(2/3)},sim)
    sim.move_to_com()

    return cm.Poincare.from_Simulation(sim)

def row_to_rebound_sim(row,npl):
    DEG2RAD = np.pi / 180
    sim = rb.Simulation()
    sim.units = ('Msun','days','AU')
    Mstar = row['mass_star_m_sun']
    sim.add(m=Mstar)
    for i in range(npl):
        h,k = np.real((row[f'h_{i}'],row[f'k_{i}']))
        ecc = np.sqrt(h*h+k*k)
        pomega = np.arctan2(h,k)
        
        sim.add(
            m = row[f'mass_planet_star_ratio_{i}'] * Mstar,
            P = row[f'period_days_{i}'],
            e = ecc,
            inc = row[f'inclination_deg_{i}'] * DEG2RAD,
            l = row[f'mean_longitude_deg_{i}'] * DEG2RAD,
            pomega = pomega,
            Omega = row[f'longitude_of_ascending_node_deg_{i}'] * DEG2RAD
        )
    sim.move_to_com()

    return sim

def tau_alphas_to_tau_as(tau_alpha, masses, resonances):
    Npl = len(masses)
    sma = np.ones(Npl)
    mtrx = -1 * np.eye(Npl)
    mtrx[:, 0] = 1
    for i, jk in enumerate(resonances):
        j, k = jk
        sma[i + 1] = sma[i] * (j / (j - k)) ** (2 / 3)
    mtrx[0] = masses / sma
    gamma_alphas = np.concatenate(([0], 1 / np.array(tau_alpha)))
    gamma_a = np.linalg.inv(mtrx) @ gamma_alphas
    return 1 / gamma_a
