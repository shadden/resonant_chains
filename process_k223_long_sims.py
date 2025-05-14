import celmech as cm
import numpy as np
import rebound as rb

file_dir = "/fs/lustre/cita/hadden/10_resonant_chains/runme/"
long_file_templates = ("k223_{}_long-run_{}.bin",
                  "k223_{}_tp_mass_1_long-run_{}.bin",
                  "k223_{}_tp_mass_2_long-run_{}.bin"
                 )   
import sys
I_IC = int(sys.argv[1])
I_TP = int(sys.argv[2])
I_LONG = int(sys.argv[3])
sa_long = rb.Simulationarchive(file_dir+long_file_templates[I_TP].format(I_IC,I_LONG))
results_long = cm.nbody_simulation_utilities.get_simarchive_integration_results(sa_long)
savedir = "/fs/lustre/cita/hadden/10_resonant_chains/k223_sim_data/"
np.savez_compressed(savedir+f"kep223_long_TP{I_TP}_IC{I_IC}_RUN{I_LONG}",**results_long)

