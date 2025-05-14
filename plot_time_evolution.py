import celmech as cm
from matplotlib import pyplot as plt
import numpy as np
import rebound as rb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from celmech.secular import LaplaceLagrangeSystem

file_dir = "/fs/lustre/cita/hadden/10_resonant_chains/runme/"
file_templates = ("k223_with_test_particles_{}.bin",
                  "k223_with_mpluto_test_particles_{}.bin",
                  "k223_with_3mpluto_test_particles_{}.bin"
                 )
long_file_templates = ("k223_{}_long-run_{}.bin",
                  "k223_{}_tp_mass_1_long-run_{}.bin",
                  "k223_{}_tp_mass_2_long-run_{}.bin"
                 )   
from resonant_chains import resonant_chain_variables_transformation_matrix
resonances = [(4,1),(3,1),(4,1)]
T = resonant_chain_variables_transformation_matrix(resonances)
import sys

I_IC = int(sys.argv[1])
I_TP = int(sys.argv[2])
I_LONG = int(sys.argv[3])
filename = file_dir+file_templates[I_TP].format(I_IC)
sa = rb.Simulationarchive(filename)
sa_long = rb.Simulationarchive(file_dir+long_file_templates[I_TP].format(I_IC,I_LONG))
results_long = cm.nbody_simulation_utilities.get_simarchive_integration_results(sa_long)

Npl = sa[0].N_active-1
Nout=4096
target_times = np.linspace(0,sa.tmax,Nout)
times,Delta,ecc,Period = np.zeros(Nout),np.zeros((Npl-1,Nout)),np.zeros((Npl,Nout)),np.zeros((Npl,Nout))
lmbda = np.zeros((Npl,Nout))
pmg = np.zeros((Npl,Nout))
Omg = np.zeros((Npl,Nout))
for i,sim in enumerate(sa.getSimulations(target_times)):
    times[i] = sim.t
    ps = sim.particles[:sim.N_active]
    ecc[:,i] = [p.e for p in ps[1:]]
    lmbda[:,i] = [p.l for p in ps[1:]]
    pmg[:,i] = [p.pomega for p in ps[1:]]
    Omg[:,i] = [p.Omega for p in ps[1:]]
    Period[:,i] = [p.P for p in ps[1:]]
    for j,pin,pout,pq in zip(range(Npl-1),ps[1:-1],ps[2:],resonances):
        p,q = pq
        Delta[j,i] = (p-q)*pout.P / pin.P / p - 1
time  = times.copy()

# Define scale and time split
P_0p1au = 0.1**(1.5)
P_code = sa[0].particles[1].P
tscale = (P_0p1au/P_code) / 1e6

t_break = tscale * time[-1]
t_long = tscale * (results_long['time'] + time[-1])
t_end = np.max(results_long['time'][np.all(results_long['a']>0,axis=0)]) * tscale



sim = rb.Simulation()
sim.add(m=1)
a_mean = np.mean(results_long['a'][:,:10_000],axis=1)
for p,a_i in zip(sa_long[0].particles[1:],a_mean):
    sim.add(m=p.m,a=a_i)
sim.move_to_com()
llsys = LaplaceLagrangeSystem.from_Simulation(sim)
llsys.add_first_order_resonance_term(1,2,resonances[0][0])
llsys.add_first_order_resonance_term(2,3,resonances[1][0])
llsys.add_first_order_resonance_term(3,4,resonances[2][0])
llsys.add_first_order_resonance_term(1,3,2)
llsys.add_first_order_resonance_term(2,4,2)

T,D = llsys.diagonalize_eccentricity()

Nout = 2048
times = np.linspace(sa_long.tmin,t_end/tscale,Nout)
xvars = np.zeros((Nout,4),dtype =complex)
for i,sim in enumerate(sa_long.getSimulations(times)):
    pvars = cm.Poincare.from_Simulation(sim)
    xvars[i] = [p.x for p in pvars.particles[1:]]
t_long_u = tscale * (time[-1] + times)

uvars = np.transpose(T.T @ xvars.T)
mean_rt_lmbda = np.mean([np.sqrt(p.Lambda) for p in pvars.particles[1:]])

Nout = 512
times_short = np.linspace(sa.tmin,sa.tmax,Nout)
xvars_short = np.zeros((Nout,4),dtype =complex)
for i,sim in enumerate(sa.getSimulations(times_short)):
    sim_c = rb.Simulation()
    for p in sim.particles[:5]:
        sim_c.add(p.copy())
    pvars = cm.Poincare.from_Simulation(sim_c)
    xvars_short[i] = [p.x for p in pvars.particles[1:5]]

uvars_short = np.transpose(T.T @ xvars_short.T)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create figure and GridSpec
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(3, 2, width_ratios=[3, 4], wspace=0.025)

# Create axes with shared x-axis within each column
axL = [fig.add_subplot(gs[0, 0])]
for i in range(1, 3):
    axL.append(fig.add_subplot(gs[i, 0], sharex=axL[0]))

axR = [fig.add_subplot(gs[0, 1])]
for i in range(1, 3):
    axR.append(fig.add_subplot(gs[i, 1], sharex=axR[0]))

# Set limits
axL[0].set_ylim(-0.0015, 0.05)
axL[1].set_ylim(0, 0.102)
axL[2].set_ylim(0, 0.035)

# Set limits
axR[0].set_ylim(-0.0015, 0.05)
axR[1].set_ylim(0, 0.102)
axR[2].set_ylim(0, 0.035)

# Set limits
axR[0].set_xlim(xmax = 17.5)

# Hide spines and manage ticks
for i, (aL, aR) in enumerate(zip(axL, axR)):
    # Only bottom subplot keeps x-tick labels
    if i < 2:
        aL.tick_params(labelbottom=False)
        aR.tick_params(labelbottom=False)
    aL.tick_params(direction='in', labelsize=12, size=6)
    aR.tick_params(direction='in', labelsize=12, size=6, labelleft=False)

# Plot data
for i in range(3):
    j, k = resonances[i]
    Delta_i = (j - k) * results_long['P'][i + 1] / results_long['P'][i] / j - 1
    l, = axL[0].plot(tscale * time, Delta[i],label=f"({i+1},{i+2})")
    axR[0].plot(t_long, Delta_i, color=l.get_color(),label=f"({i+1},{i+2})")
axL[0].legend(title = "Planet pair",loc='upper right')
for i in range(4):
    ecc_i = results_long['e'][i]
    l, = axL[1].plot(tscale * time, ecc[i],label=f"{i+1}")
    axR[1].plot(t_long, ecc_i, color=l.get_color())
axL[1].legend(title="Planet")

axL[2].plot(times_short * tscale , np.abs(uvars_short) / mean_rt_lmbda)
axR[2].plot(t_long_u, np.abs(uvars) / mean_rt_lmbda)
axL[2].legend([i for i in range(1,5)],title="Mode #")
# Labels
axL[0].set_ylabel(r"$\Delta$", fontsize=16)
axL[1].set_ylabel(r"$e$", fontsize=16)
axL[2].set_ylabel(r"$|u|/\langle{\Lambda}^{1/2}\rangle$", fontsize=16)
axL[2].set_xlabel("Time [Myr]", fontsize=16)
axR[2].set_xlabel("Time [Myr]", fontsize=16)


# Set x-axis limits
for a in axL:
    a.set_xlim(0, t_break)
for a in axR:
    a.set_xlim(t_break + 1e-6, t_end)

# Adjust vertical space between plots
plt.subplots_adjust(hspace=0.05)
plt.tight_layout()
plt.savefig(f"/cita/h/home-2/hadden/Projects/21_ResonantChains/figs/kep223_IC{I_IC}_TP{I_TP}_SIM{I_LONG}.png")
