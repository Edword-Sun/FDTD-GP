import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light (m/s)
dx = 0.01  # Spatial step size (m)
dt = dx / c  # Temporal step size (s)
T = 100  # Total simulation time steps

# Grid parameters
nx = 100  # Number of grid points in x direction
ny = 100  # Number of grid points in y direction
nz = 100  # Number of grid points in z direction

# Initialize electric field components
Ex = np.zeros((nx, ny, nz))
Ey = np.zeros((nx, ny, nz))
Ez = np.zeros((nx, ny, nz))

# Initialize magnetic field components
Hx = np.zeros((nx, ny, nz))
Hy = np.zeros((nx, ny, nz))
Hz = np.zeros((nx, ny, nz))

# Main loop for FDTD simulation
for t in range(T):
    # Update electric field components
    Ex[1:, :, :] += dt / dx * (Hz[1:, :, :] - Hz[:-1, :, :])
    Ey[:, 1:, :] -= dt / dx * (Hz[:, 1:, :] - Hz[:, :-1, :])
    Ez[:, :, 1:] += dt / dx * (Hy[:, :, 1:] - Hy[:, :, :-1])

    # Update magnetic field components
    Hx[:-1, :, :] -= dt / dx * (Ez[1:, :, :] - Ez[:-1, :, :])
    Hy[:, :-1, :] += dt / dx * (Ez[:, 1:, :] - Ez[:, :-1, :])
    Hz[:, :, :-1] -= dt / dx * (Ey[:, :, 1:] - Ey[:, :, :-1])

    # Source function (excitation)
    Ex[nx//2, ny//2, nz//2] += np.sin(2 * np.pi * 500e6 * t * dt)

    # Visualization (optional)
    if t % 10 == 0:
        plt.imshow(np.sqrt(Ex[:, :, nz//2]**2 + Ey[:, :, nz//2]**2 + Ez[:, :, nz//2]**2), cmap='inferno', origin='lower')
        plt.colorbar()
        plt.title(f"Time Step: {t}")
        plt.pause(0.01)
        plt.clf()

# Display final result
plt.imshow(np.sqrt(Ex[:, :, nz//2]**2 + Ey[:, :, nz//2]**2 + Ez[:, :, nz//2]**2), cmap='inferno', origin='lower')
plt.colorbar()
plt.title("Final Electric Field Magnitude (z-plane)")
plt.show()
