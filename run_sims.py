import rebound
import os
import sys

def integrate_simulation(NUM, I, time_limit=1e8,Nout = 2**14):
    # Define the filename pattern
    filename = f"./kep223_files/kep223-f{NUM}_{I}.bin"
    
    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
    # Load the simulation from the binary file
    sim = rebound.Simulation(filename)
    
    # Modify integrator settings
    P1 = sim.particles[1].P
    sim.integrator = 'whfast'
    sim.dt = P1 / 25.
    sim.ri_whfast.safe_mode = 0
    Tfin = time_limit * P1
    dt_out = Tfin / Nout
    Nsteps_out = int(dt_out / sim.dt)
    print(f"Saving output every {Nsteps_out}",flush=True)
    # Create a SimulationArchive to store snapshots
    sim.save_to_file(f"archive_kep223-f{NUM}_{I}.sa",step = Nsteps_out,delete_file = True)
    sim.integrate(Tfin)

def main():
    if len(sys.argv) != 3:
        print("Usage: python integrate_simulation.py <NUM> <I>")
        sys.exit(1)
    
    # Parse the command line arguments
    NUM = float(sys.argv[1])  # NUM is a float
    I = int(sys.argv[2])      # I is an integer
    
    try:
        # Integrate the simulation and get the archive
        integrate_simulation(NUM, I,1e6,512)
        
        # Optionally save the archive to a file
        print(f"Simulation complete. Archive saved as 'archive_kep223-f{NUM}_{I}.sa'.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
