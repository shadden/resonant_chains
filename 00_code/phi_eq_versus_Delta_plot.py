from matplotlib import pyplot as plt
import numpy as np
import pickle
from resonant_chains import resonant_chain_variables_transformation_matrix
from pathlib import Path
import pandas as pd
from scipy.stats import norm
n=1
LOWER, UPPER = norm.cdf([-n, n])
do_wrap = False
if do_wrap:
    wrap = lambda x: np.mod(x+180,360)-180
else:
    wrap = lambda x: np.mod(x,360)

def ttv_samples_to_dataframe(samples):
    """
    Convert TTV posterior samples into a pandas DataFrame.

    Parameters
    ----------
    samples : ndarray, shape (N, 26)
        Array of posterior samples. Columns are:
        [M_star,
         M1, P1, t01, h1, k1,
         M2, P2, t02, h2, k2,
         ...
         M5, P5, t05, h5, k5]

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with appropriately labeled columns.
    """
    n_cols = samples.shape[1]
    
    if (n_cols - 1) % 5 != 0:
        raise ValueError("Unexpected number of columns; expected 1 + 5*N_planets")

    n_planets = (n_cols - 1) // 5

    columns = ["M_star"]

    for i in range(1, n_planets + 1):
        columns.extend([
            f"M{i}",
            f"P{i}",
            f"t0_{i}",
            f"h{i}",
            f"k{i}",
        ])

    df = pd.DataFrame(samples, columns=columns)
    return df

def main():

    ############### Kepler-80 ###############
    resonances_kepler80 = [[3,1],[3,1],[4,1],[3,1]]
    kep_80_phi_eq_K2eq0 = wrap(np.array([-150.12334743,  -67.02201052,  -45.46753898]))
    # load Kepler-80 posterior data
    parent_dir = Path("/Users/hadden/Papers/10_chain_dynamics/03_data") 
    file_str = parent_dir / "Kepler-80-full" / "trimmed_ecc_yesg.pkl"
    with open(file_str, "rb") as f:
        data = pickle.load(f)

    # convert to DataFrame
    weisserman_samples = ttv_samples_to_dataframe(data)

    # compute resonant angles for each sample
    epoch = 0
    for i in range(1,6):
        t0 = weisserman_samples[f't0_{i}']
        P = weisserman_samples[f'P{i}']
        weisserman_samples[f'lambda_{i}'] = np.mod(2 * np.pi * (epoch - t0) / P,2*np.pi) + 0.5 * np.pi

    
    T_weiss = resonant_chain_variables_transformation_matrix(resonances_kepler80   ,max_order=1)[2:5,:5]
    phi_weiss = T_weiss @ [weisserman_samples[f'lambda_{i}'] for i in range(1,6)]

    fname_str = "/Users/hadden/Papers/10_chain_dynamics/03_data/Kepler-80-full/kep80eqs/kep-80-phi-eq-{}.npz"
    file0 = np.load(fname_str.format(1))
    Delta_in_kep80 = file0['Delta_in']
    Delta_obs_kep80 = file0['Delta_obs']
    kep80_data = {}
    for key in file0.keys():
        if key != 'Delta_obs':
            kep80_data.update({key:np.concatenate([np.load(fname_str.format(i))[key] for i in range(1,11)])})

    ############### TOI-178 ###############

    resonances_toi178 = [[2,1],[3,1],[3,1],[4,1]]
    toi178_phi_eq_K2eq0 = np.array([180.52205252, 158.11017074,  73.32144125])
    leleu_samples = pd.read_pickle("/Users/hadden/Papers/10_chain_dynamics/03_data/TOI-178/leleu_m_uniform_e_loguniform_samples.pkl")
    lambdas = np.array([leleu_samples[f'mean_longitude_deg_{i}'] for i in range(5)])
    T_leleu = resonant_chain_variables_transformation_matrix(resonances_toi178,max_order=1)[2:5,:5]
    leleu_phi = T_leleu @ lambdas

    fname_str = "/Users/hadden/Papers/10_chain_dynamics/03_data/TOI-178/toi178eqs/toi-178-phi-eq-{}.npz"
    file0 = np.load(fname_str.format(1))
    Delta_in_toi178 = file0['Delta_in']
    Delta_obs_toi178 = file0['Delta_obs']
    toi178_data = {}
    for key in file0.keys():
        if key != 'Delta_obs':
            toi178_data.update({key:np.concatenate([np.load(fname_str.format(i))[key] for i in range(1,11)])})


    ############### plot ###############

    fig,ax = plt.subplots(3,2,figsize=(12,6),sharex='col')
    
    for i in range(3):
        
        phi_eq = kep_80_phi_eq_K2eq0[i]
        ylo,ymed,yhi  = np.quantile(wrap(180 * kep80_data['phi_eqs'][:,:,i]/np.pi),[LOWER,0.5,UPPER],axis=0)
        ax[i,0].fill_between(
            Delta_in_kep80,
            ylo-phi_eq,
            yhi-phi_eq,
            color = 'k',alpha=0.25
        )
        ax[i,0].plot(Delta_in_kep80,ymed-phi_eq,color = 'k')
        yobs_low,yobs_med,yobs_hi = np.quantile(wrap(phi_weiss[i]*180/np.pi),[LOWER,0.5,UPPER])
        ax[i,0].errorbar(Delta_obs_kep80,yobs_med-phi_eq,yerr = [[yobs_med-yobs_low],[yobs_hi-yobs_med]],color='red',marker='s',
                         label = "Weisserman+2023")
        ax[i,0].set_ylim(-11,11)
        ax[i,0].set_yticks([-10,-5,0,5,10])
        ax[i,0].tick_params(direction='in',size=6,labelsize=14)
        ax[i,0].set_ylabel(r"$\delta\phi_{{{0}}}$ [$^\circ$]".format(i+1),fontsize=12) #- \phi_{{{0},eq}}(K_2=0)
        k1,k2,k3 = T_weiss[i][i:i+3].astype(int)
        ax[i,0].text(
            0.95,0.2,
            fr"$({k3:d}\lambda_{i+3} - {-1*k2:d}\lambda_{i+2} + {k1:d}\lambda_{i+1})_{{eq}} = {phi_eq:.0f}^{{\circ}}$",
            transform=ax[i,0].transAxes,
            fontsize=12,
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
        if i==0:
            ax[i,0].legend(loc='upper left',fontsize=12)
    ax[2,0].set_xlabel(r"$\Delta_{1,2}$",fontsize=16)
    ax[0,0].set_title("Kepler-80",fontsize=16)

    for i in range(3):
        ylo,ymed,yhi  = np.quantile(wrap(180 * toi178_data['phi_eqs'][:,:,i]/np.pi),[LOWER,0.5,UPPER],axis=0)
        phi_eq = toi178_phi_eq_K2eq0[i]
        ax[i,1].fill_between(
            Delta_in_toi178,
            ylo-phi_eq,
            yhi-phi_eq,
            color = 'k',alpha=0.25
        )
        ax[i,1].plot(Delta_in_toi178,ymed-phi_eq,color = 'k')
        yobs_low,yobs_med,yobs_hi = np.quantile(wrap(leleu_phi[i]),[LOWER,0.5,UPPER])
        ax[i,1].errorbar(
            Delta_obs_toi178,yobs_med-phi_eq,
            yerr = [[yobs_med-yobs_low],[yobs_hi-yobs_med]],color='red',marker='s',
            label = "Leleu+2024"
        )
        k1,k2,k3 = T_leleu[i][i:i+3].astype(int)
        ax[i,1].text(
            0.95,0.8 if i!=1 else 0.2,
            fr"$({k3:d}\lambda_{i+3} - {-1*k2:d}\lambda_{i+2} + {k1:d}\lambda_{i+1})_{{eq}} = {phi_eq:.0f}^{{\circ}}$",
            transform=ax[i,1].transAxes,
            fontsize=12,
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
        ax[i,1].tick_params(direction='in',size=6,labelsize=14)
        if i==0:
            ax[i,1].legend(loc='upper left',fontsize=12)
            ax[i,1].set_ylim(-17.5,17.5)
            ax[i,1].set_yticks([-15,-10,-5,0,5,10,15])
        else:
            ax[i,1].set_ylim(-11,11)
            ax[i,1].set_yticks([-10,-5,0,5,10])

    ax[2,1].set_xlabel(r"$\Delta_{1,2}$",fontsize=16)
    ax[0,1].set_title("TOI-178",fontsize=16)

    plt.show()


if __name__ == "__main__":
    main()

