import rebound                                                                                                                                                             
import numpy as np
import rebound as rb
import sys

I = int(sys.argv[1])
J = int(sys.argv[2])
sim_name = "k223_with_test_particles_{}.bin".format(I)
sim0 = rb.Simulation(sim_name)
P0 = sim0.particles[1].P
sim0.integrate(J*100*P0)

sim = rb.Simulation()
for p in sim0.particles[:sim0.N_active]:
    sim.add(p.copy())
sim.integrator = 'whfast'
sim.dt = P0 / 25.
sim.ri_whfast.safe_mode = 0

