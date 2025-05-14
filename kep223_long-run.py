import numpy as np
import rebound as rb
from celmech.nbody_simulation_utilities import align_simulation, set_time_step
import sys
I = int(sys.argv[1])
J = int(sys.argv[2])

# Load simulation
savedir = "/fs/lustre/cita/hadden/10_resonant_chains/runme/"
simfile = savedir+f"k223_with_test_particles_{I}.bin"
sim0 = rb.Simulation(simfile)

# advance initial simulation
P0 = sim0.particles[1].P
sim0.integrate(J*50*P0)

# copy active particles
sim = rb.Simulation()
for p in sim0.particles[:sim0.N_active]:
    sim.add(p.copy())
sim.move_to_com()
sim.integrator='whfast'
sim.ri_whfast.safe_mode = 0
align_simulation(sim)
set_time_step(sim,1/25.)

# set integration parameters
savefile = savedir+f"k223_{I}_long-run_{J}.bin"
dt_out_interval = 1e6 * P0 / 128
n_steps_out = int(np.floor(dt_out_interval / sim.dt))
sim.save_to_file(savefile,step=n_steps_out)
sim.integrate(1e9 * P0)

