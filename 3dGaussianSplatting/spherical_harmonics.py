import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

def plot_spherical_harmonic(l, m):
    """
    Plots the spherical harmonic Y_l^m(theta, phi).
    
    Parameters:
    l (int): Degree of the spherical harmonic
    m (int): Order of the spherical harmonic
    """
    
    # Create a grid of theta (0 to pi) and phi (0 to 2pi)
    theta = np.linspace(0, np.pi, 100)  # Polar angle (theta)
    phi = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle (phi)
    
    # Create a meshgrid of theta and phi values
    theta, phi = np.meshgrid(theta, phi)
    
    # Calculate the spherical harmonic
    Y = sph_harm(m, l, phi, theta)  # Y(l, m) = spherical harmonic function
    
    # Convert from spherical to Cartesian coordinates
    x = np.abs(Y) * np.sin(theta) * np.cos(phi)
    y = np.abs(Y) * np.sin(theta) * np.sin(phi)
    z = np.abs(Y) * np.cos(theta)
    
    # Plotting the spherical harmonic
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='b', alpha=0.6)
    
    # Setting the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_title(f"Spherical Harmonic Y_{l}^{m}(θ, φ)")
    plt.show()

# Example: Plotting Y_3^2
plot_spherical_harmonic(3, 2)
