import rebound as rb
import numpy as np
from celmech.nbody_simulation_utilities import add_canonical_heliocentric_elements_particle
import rebound as rb
import celmech as cm

def damping_rate_tides(masses,resonances,radii, Qs = 100):
    """
    Compute the tidal damping timescales for a series of resonant planets.
    Parameters
    ----------
    masses : list of floats
        Masses of the planets in units of the stellar mass.
    resonances : list of tuples
        List of tuples (p, q) representing the resonances between adjacent planets.
    radii : list of floats
        Radii of the planets in units of inner planet's semi-major axis.
    Qs : float or list of floats, optional
        Tidal quality factors for the planets. If a single float is provided, it will be applied to all planets. Default is 100.
    """
    if isinstance(Qs, (int, float)):
        Qs = [Qs] * len(masses)
    
    periods = np.cumprod(np.array([1] + list(map(lambda x: x[0]/(x[0]-x[1]),resonances))))
    smas = periods**(2/3)
    taus = (4/63) * (smas / radii)**5 * Qs * masses  * periods / (2 * np.pi)
    return 1/taus


def relative_damping_rate_tau_wave(masses,resonances,alpha = 1,beta = 0.25):
    """
    Compute the wave timescales for a series of resonant planets.
    Parameters
    ----------
    masses : list of floats
        Masses of the planets in units of the stellar mass.
    resonances : list of tuples
        List of tuples (p, q) representing the resonances between adjacent planets.
    alpha : float, optional
        Power-law index for the semi-major axis dependence of the surface density profile. Default is is 1.
    beta : float, optional
        Power-law index for the semi-major axis dependence of the disk flaring (H/R ~ a^beta). Default is 0.25.
    """
    taus = 1 / masses
    periods = np.cumprod(np.array([1] + list(map(lambda x: x[0]/(x[0]-x[1]),resonances))))
    smas = periods**(2/3)
    sigma_asq_scaling = smas ** (2-alpha)
    h_scaling = smas ** (beta)
    taus = periods * taus  * h_scaling**4 / sigma_asq_scaling
    gamma_rel = taus[0] / taus 
    return gamma_rel

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
