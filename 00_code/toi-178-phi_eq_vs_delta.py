import numpy as np
from three_body_resonance_eq import get_three_br_eq_angles


system_name = "TOI-178"
resonances = [[2,1],[3,1],[3,1],[4,1]]
phi_guess = np.array([180.0,161.49371826  ,70.72549925]) * np.pi / 180
Delta_in_obs = 0.01256
print(f"Observed Delta: {Delta_in_obs:.5f}")

Nmasses =  3
NDeltas = 3
Delta_in_vals = np.geomspace(0.5,2,NDeltas) * Delta_in_obs
phi_solns = np.zeros((Nmasses,NDeltas,len(resonances) - 1))
masses_done = np.zeros((Nmasses, len(resonances)+1))

leleu_mass_samples = np.load("/Users/hadden/Papers/10_chain_dynamics/03_data/TOI-178/leleu_masses.npy")

for i in range(Nmasses):
    indx = np.random.randint(len(leleu_mass_samples))
    print(i,indx)
    masses_done[i] = leleu_mass_samples[indx] * 3e-6
    for j,Delta_in in enumerate(Delta_in_vals):
        soln = get_three_br_eq_angles(
            masses_done[i],
            resonances,
            Delta_in,
            phi_guess = phi_guess
        )
        phi_guess = soln
        phi_solns[i,j] = soln
        