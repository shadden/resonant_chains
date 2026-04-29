import numpy as np
import rebound as rb
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
from resonantstate.data_download  import get_metadata_observations, download_observations_samples
from resonantstate.analyse_samples import *
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
_preffered_author_id = {
    "TOI-178" : 1,
    "Kepler-60": 3,
    "Kepler-223" : 1
}
data_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") 

def dataframe_row_to_rebound_sim(row,npl,delta_i_safety = 0.01):
    """_summary_

    Parameters
    ----------
    row : pandas.Series 
        Dataframe row containing the system parameters
    npl : int
        Number of planets
    delta_i_safety : float, optional
        Safety margin for inclination to avoid coordinate issues at i=90°, by default 0.01

    Returns
    -------
    rb.Simulation 
        Rebound simulation initialized with the system parameters
    """
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
            inc = (row[f'inclination_deg_{i}'] - delta_i_safety)* DEG2RAD,
            l = row[f'mean_longitude_deg_{i}'] * DEG2RAD,
            pomega = pomega,
            Omega = row[f'longitude_of_ascending_node_deg_{i}'] * DEG2RAD
        )
    sim.move_to_com()

    return sim
from resonant_chains import resonant_chain_variables_transformation_matrix
def compute_three_body_angle_time_series(sim,times,resonances):
    """
    Compute timeseries of three-body resonant angles

    Parameters
    ----------
    sim : rebound.Simulation
        Simulation to integrate
    times : 1d array
        Times to compute three-body angles at
    resonances : list
        List of integers specifying resonances of adjacent planets

    Returns
    -------
    ndarray
        three-body angles of each adjacent trio of planets at each simulation time.
    """
    Nout = len(times)
    Npl = len(sim.particles) - 1
    N3br = len(resonances) - 1
    angles = np.zeros((Nout,len(resonances)-1))
    
    T = resonant_chain_variables_transformation_matrix(resonances,1)
    T = T[2:2+N3br,:Npl]
    print(T)
    for i,t in enumerate(times):
        sim.integrate(t)
        mean_longitudes = np.array([p.l for p in sim.particles[1:]])
        angles[i,:] = T @ mean_longitudes
    angles = np.mod(angles,2*np.pi)
    return angles

def fmft_reconstruct_signal(times,freq_amp_dict):
    """
    Reconstruct signal from frequency-amplitude dictionary obtained from FMFT

    Parameters
    ----------
    times : 1d array
        Times to reconstruct signal at
    freq_amp_dict : dict
        Dictionary with frequencies as keys and complex amplitudes as values

    Returns
    -------
    1d array
        Reconstructed signal at given times
    """
    signal = np.zeros_like(times, dtype=np.complex128)
    for freq, amp in freq_amp_dict.items():
        signal += amp * np.exp(1j * freq * times)
    return signal.real
def compute_angle_frequencies(times,angles):
    """
    Compute frequencies of three-body angles using FMFT

    Parameters
    ----------
    times : 1d array
        Times corresponding to angle measurements
    angles : 2d array
        Three-body angles at each time (shape: [N_times, N_angles])

    Returns
    -------
    list of dicts
        List of frequency-amplitude dictionaries for each angle
    """
    freq_amp_dicts = []
    for i in range(angles.shape[1]):
        freq_amp_dict = fmft(times,angles[:,i] - np.mean(angles[:,i]),2*(angles.shape[1]))
        freq_amp_dicts.append(freq_amp_dict)
    return freq_amp_dicts

def main():
    parser = ArgumentParser(description="Compute three-body angles for a resonant chain and apply FMFT")
    parser.add_argument(
        "--system", 
        type=str, 
        required=True, 
        help="Name of the system to analyze (e.g., 'TOI-178')"
    )
    parser.add_argument(
        "--i_start",
        type=int,
        default=0,
        help="Starting index for the analysis"
    )
    parser.add_argument(
        "--i_stop",
        type=int,
        default=10,
        help="Stopping index for the analysis"
    )
    args = parser.parse_args()
    system_name = args.system
    i_start = args.i_start
    i_stop = args.i_stop

    # load pandas data on system
    df_all_obs = get_metadata_observations()
    df_selected = df_all_obs[df_all_obs["star_name"].isin([system_name])]

    print(f"Downloading samples for {system_name}")
    data_list = download_observations_samples(df_selected,None)
    data = data_list[_preffered_author_id[system_name]]
    Nplanets = data['nb_planets']
    df = data['samples']

    # load cached dynamics data
    dynamics_data = np.load(f"/Users/hadden/Papers/10_chain_dynamics/03_data/{system_name}/{system_name}_maxorder3.npz")
    resonances = dynamics_data['resonances']

    all_fmft_results = []
    for i in range(i_start,i_stop):
        # compute three-body angles time series
        sim = dataframe_row_to_rebound_sim(df.iloc[i],Nplanets)
        sim.integrator = "whfast"
        sim.dt = sim.particles[1].P / 25
        times = np.linspace(0, 5e4 * sim.particles[1].P, 512)

        angles = compute_three_body_angle_time_series(sim,times,resonances)
        angles = 180*angles/np.pi 


        freq_amp_dicts = compute_angle_frequencies(times,angles)        
        all_fmft_results.append(freq_amp_dicts)
        
    # Save results to a file
    pickle_path = data_dir / system_name / f"{system_name}_three_body_angles_fmft_{i_start}_{i_stop}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_fmft_results, f)
    print(f"Saved FMFT results to {pickle_path}")
if __name__ == "__main__":
    main()