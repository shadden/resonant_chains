import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import SymLogNorm
import matplotlib.gridspec as gridspec
sys.path.insert(0,"../00_code/")
from pathlib import Path
import argparse
from celmech.disturbing_function import list_resonance_terms
from resonant_chains import get_chain_rebound_sim, get_chain_hpert, ResonantChainPoissonSeries
from scipy.optimize import linear_sum_assignment
import sympy as sp
from scipy.interpolate import interp1d
data_path = "/Users/hadden/Papers/10_chain_dynamics/03_data/"

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute damping rate coefficient for resonant chain equilibria"
    )
    p.add_argument(
        "--system",
        default="Kepler-60",
        help="System to analyze"
    )
    p.add_argument(
        "--plot_e",
        action='store_true',
        help="Plot individual planet eccentricities in first panel."
    )
    p.add_argument(
        "--plot_obs",
        action='store_true',
        help="Plot observed |Z| values from TTV data."
    )

    p.add_argument(
        "--save-plot",
        default="",
        help="If provided, save the plot to this path instead of showing it (e.g., plot.png or plot.pdf)."
    )

    return p.parse_args()

def main():
    args = parse_args()
    system = args.system
    diss_rate_path = data_path+f"{system}/{system}_dissipation_info.npz"
    z_data_path = data_path+f"{system}/pairZs.npy.npz"
    diss_rate_data = np.load(diss_rate_path)
    data = np.load(data_path+f"{system}/elliptic_eq_data.npz")
    gamma_e_coeffs = diss_rate_data["damping_rate_coeffs"]

    freqs = data['freqs']
    dK2vals = 1e3 * data['dK2vals']
    eccs = data['eccs']
    Zs = data['Z']
    kappa2_dot = data['kappa2_dot']
    N_dof = freqs.shape[1]
    Npl = N_dof//2 + 1
    N_fast = Npl - 1

    
    z_data = np.load(z_data_path) if Path(z_data_path).exists() else None
        
    # Set up the figure and gridspec
    fig = plt.figure(figsize=((N_fast + 2) * 3, 6))  # Extra width for colorbar
    gs = gridspec.GridSpec(1, N_fast + 3, width_ratios=[1] * 2 + [0.7]*N_fast + [0.1], wspace=0.12)


    # Colormap and normalization
    cmap = plt.get_cmap('RdBu')
    ymin, ymax = np.min(dK2vals), np.max(dK2vals)
    xmin, xmax = 0.5, Npl + 0.5

    linthresh = np.percentile(np.abs(gamma_e_coeffs), 34)
    maxabs = np.max(np.abs(gamma_e_coeffs))
    norm = SymLogNorm(linthresh=linthresh, vmin=-maxabs, vmax=maxabs)

    axes = []
    im = None
    
    # First panel: line plot
    ax0 = fig.add_subplot(gs[0])
    dK2_obs_inferred_and_line_color = []
    for l,Z_i in enumerate(Zs.T):
        line,=ax0.plot(Z_i, dK2vals,label = "$| Z_{{{},{}}}|$".format(l+1,l+2),ls='-',lw=3)
        if z_data and args.plot_obs:
            if system == "TOI-1136" and l>0:
                print("hello")
                continue
            try:
                xpts = z_data[f"{l+1},{l+2}"]
            except KeyError:
                continue
            ypts = interp1d(Z_i,dK2vals,fill_value="extrapolate")(z_data[f"{l+1},{l+2}"])
            xlo,xmed,xhi =  xpts[1:4]
            ylo,ymed,yhi =  np.sort(ypts[1:4])
            print([[xmed - xlo], [xhi - xmed]])
            line_color = line.get_color()
            plt.errorbar(
                xmed, ymed,
                xerr=[[xmed - xlo], [xhi - xmed]],
                yerr=[[ymed - ylo], [yhi - ymed]],
                marker = 'o',
                ms = 10,
                color = line_color
            )
            dK2_obs_inferred_and_line_color.append((ymed,line_color))
    # plot e if flag is given
    if args.plot_e:
        for l,e_i in enumerate(eccs.T):
            ax0.plot(e_i, dK2vals,label = "$e_{{{}}}$".format(l+1),ls='--',lw=1)

    ax0.set_title("Equlibirum eccentricity",fontsize=15)
    ax0.set_ylabel(r"$10^3 \times K_2$",fontsize=15)
    ax0.set_xlabel(r"$e$",fontsize=15)
    emax = np.max(eccs)
    xti_max = int(np.ceil(emax / 0.02))
    xts = [0.02 * ix for ix in range(0,xti_max + 1)]
    ax0.set_xticks(xts,labels = [f"{x:g}" for x in xts] )
    ax0.tick_params(direction='in', size=6, labelsize=12)
    ax0.legend(loc='upper left')
    # Add observations legend
    if args.plot_obs:
        obs_handle = Line2D(
            [], [], 
            marker='o',
            ms = 10,
            color='black', 
            linestyle='None', 
            label='Observations'
        )
        handles, labels = ax0.get_legend_handles_labels()
        handles.append(obs_handle)
        labels.append('Observations')
        ax0.legend(handles, labels)

    axes.append(ax0)
    ax1 = fig.add_subplot(gs[1],sharey=ax0)
    ax1.tick_params(labelleft=False)
    colors = []
    for l,omega_i in enumerate(freqs.T):
        if l < Npl - 2:
            l,=ax1.plot(omega_i, dK2vals,label = r"$\omega_{{{}}}$".format(l+1),lw=2)
        else:
            l,=ax1.plot(omega_i + kappa2_dot, dK2vals,label = r"$\omega_{{{}}} + \dot{{\theta}}_{{{},{}}}$".format(l+1,Npl-1,Npl),lw=2)
        colors.append(l.get_color())
    ax1.legend(loc='upper left') 
    ax1.set_xlabel(r"$\omega_i / n_1$",fontsize=15)
    ax1.tick_params(direction='in', size=6, labelsize=12)
    ax1.set_xscale('log')
    ax1.set_title("Mode frequencies",fontsize=15)
    # Other panels: image plots
    last_freqs = freqs[-1,1:] + kappa2_dot[-1] 
    fast_freq_column_ids = 1 + np.argsort(last_freqs)[-N_fast:]
    for i,col_id in enumerate(fast_freq_column_ids):
        ax = fig.add_subplot(gs[i + 2],sharey=ax0)
        gamma_data = gamma_e_coeffs[:,col_id]
        col = colors[col_id]
        im = ax.imshow(
            gamma_data,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            interpolation='none',
            origin='upper',
            extent=[xmin, xmax, ymin, ymax]
        )
        for dK2,line_col in dK2_obs_inferred_and_line_color:
            ax.axhline(dK2,color=line_col,ls='--')
        
        ax.set_title(f"Mode {col_id+1} ($c_{{{col_id+1}j}}$)",color = col,fontsize=15)
        ax.set_xlabel("$j$",fontsize=15)
        ax.set_xticks(range(1,Npl+1))
        ax.tick_params(direction='in', size=6, labelsize=12)
        ax.tick_params(labelleft=False)
        axes.append(ax)
        
    # Colorbar in the last column of the GridSpec
    cax = fig.add_subplot(gs[-1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"$c_{ij}$", fontsize=15, labelpad=10, rotation=270)
    if args.save_plot:
        save_path = args.save_plot
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[info] Saved figure to {save_path}")
    else:
        plt.show()

if __name__ =="__main__":
    main()