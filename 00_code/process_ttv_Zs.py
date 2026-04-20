import numpy as np
from resonantstate.data_download  import get_metadata_observations, download_observations_samples
from celmech.disturbing_function import get_fg_coefficients
from scipy.stats import norm
from pathlib import Path
system_preffer_id = {
    "Kepler-60":3,
    "TOI-178":1,
    "Kepler-223":1,
    "TOI-1136":0
}

df_all_obs = get_metadata_observations()
for system_name,pid in system_preffer_id.items():
    print(f"System: {system_name}")
    df_selected = df_all_obs[df_all_obs["star_name"].isin([system_name])]
    download_destination_path = None
    df_list = download_observations_samples(df_selected, download_destination_path)
    df_list_preffered = df_list[pid] 
    df = df_list_preffered['samples']
    periods = np.array([df[f'period_days_{i}'].median() for i in range(df_list_preffered['nb_planets'])])
    resonances = [(j,2) if j%2 else (j//2,1) for j in np.round(2+2/(periods[1:]/periods[:-1]-1)).astype(int)]
    pair_zs = {}
    cache_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") / system_name
    for i,(j,k) in enumerate(resonances):
        f,g = get_fg_coefficients(j,k)
        z = f * df[f'k_{i}'] + 1j * f * df[f'h_{i}']
        z +=g * df[f'k_{i+1}'] + 1j * g * df[f'h_{i+1}']
        z /= np.sqrt(f*f+g+g)
        absz = np.abs(z)
        quantiles = np.quantile(absz,np.sort(np.append([0.5],norm.cdf([-2,-1,1,2]))))
        pair_zs.update({f"{i+1},{i+2}":quantiles})
    np.savez(cache_dir / "pairZs.npy",**pair_zs)