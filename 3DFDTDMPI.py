from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
c = 3e8  # Speed of light (m/s)
dx = 0.01  # Spatial step size (m)
dt = dx / c  # Temporal step size (s)
T = 100  # Total simulation time steps

# Grid parameters
nx = 100  # Number of grid points in x direction
ny = 100  # Number of grid points in y direction
nz = 100  # Number of grid points in z direction

# Initialize electric field components for each process
Ex_local = np.zeros((nx // size, ny, nz))
Ey_local = np.zeros((nx // size, ny, nz))
Ez_local = np.zeros((nx // size, ny, nz))

# Main loop for FDTD simulation
for t in range(T):
    # Update electric field components
    Ex_local[1:, :, :] += dt / dx * (Ez_local[1:, :, :] - Ez_local[:-1, :, :])
    Ey_local[:, 1:, :] -= dt / dx * (Ez_local[:, 1:, :] - Ez_local[:, :-1, :])
    Ez_local[:, :, 1:] += dt / dx * (Ey_local[:, :, 1:] - Ey_local[:, :, :-1])

    # Allgather to collect data from all processes
    Ex_global = np.zeros((nx, ny, nz))
    Ey_global = np.zeros((nx, ny, nz))
    Ez_global = np.zeros((nx, ny, nz))
    comm.Allgather([Ex_local, MPI.DOUBLE], [Ex_global, MPI.DOUBLE])
    comm.Allgather([Ey_local, MPI.DOUBLE], [Ey_global, MPI.DOUBLE])
    comm.Allgather([Ez_local, MPI.DOUBLE], [Ez_global, MPI.DOUBLE])

    # Update magnetic field components (global)
    Ex_global[:-1, :, :] -= dt / dx * (Ez_global[1:, :, :] - Ez_global[:-1, :, :])
    Ey_global[:, :-1, :] += dt / dx * (Ez_global[:, 1:, :] - Ez_global[:, :-1, :])
    Ez_global[:, :, :-1] -= dt / dx * (Ey_global[:, :, 1:] - Ey_global[:, :, :-1])

    # Source function (excitation)
    Ex_global[nx//2, ny//2, nz//2] += np.sin(2 * np.pi * 500e6 * t * dt)

    # Visualization (optional)
    if rank == 0 and t % 10 == 0:
        plt.imshow(np.sqrt(Ex_global[:, :, nz//2]**2 + Ey_global[:, :, nz//2]**2 + Ez_global[:, :, nz//2]**2), cmap='inferno', origin='lower')
        plt.colorbar()
        plt.title(f"Time Step: {t}")
        plt.pause(0.01)
        plt.clf()

# Display final result
if rank == 0:
    plt.imshow(np.sqrt(Ex_global[:, :, nz//2]**2 + Ey_global[:, :, nz//2]**2 + Ez_global[:, :, nz//2]**2), cmap='inferno', origin='lower')
    plt.colorbar()
    plt.title("Final Electric Field Magnitude (z-plane)")
    plt.show()
