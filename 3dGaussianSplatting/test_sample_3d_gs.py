import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# # Step 1: Generate random 3D points
# num_points = 100
# points_3d = np.random.randn(num_points, 3)  # Random points (x, y, z)
# colors = np.random.rand(num_points, 3)     # Random RGB colors
# weights = np.random.rand(num_points)       # Random weights (0 to 1)

# # Step 2: Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot with colors and transparency based on weights
# for i in range(num_points):
#     ax.scatter(
#         points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 
#         color=colors[i], 
#         s=weights[i] * 200,  # Scale size based on weights
#         alpha=weights[i]     # Transparency based on weights
#     )

# # Step 3: Set plot labels and limits
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("3D Gaussian Splatting Visualization")

# plt.show()

# __________________________________________________________________________


# Step 1: Generate random 3D points
# num_points = 100
# points_3d = np.random.randn(num_points, 3)  # Random points (x, y, z)
# colors = np.random.rand(num_points, 3)     # Random RGB colors
# weights = np.random.rand(num_points)       # Random weights (0 to 1)

# def gaussian_3d(x, y, z, mean, sigma):
#     """3D Gaussian function."""
#     return np.exp(-((x - mean[0])**2 + (y - mean[1])**2 + (z - mean[2])**2) / (2 * sigma**2))

# # Step 1: Create a grid of points in 3D space
# grid_size = 50
# x = np.linspace(-2, 2, grid_size)
# y = np.linspace(-2, 2, grid_size)
# z = np.linspace(-2, 2, grid_size)
# X, Y, Z = np.meshgrid(x, y, z)

# # Step 2: Compute Gaussian values
# sigma = 0.5  # Gaussian spread
# gaussian_splats = np.zeros(X.shape)

# for point in points_3d:
#     gaussian_splats += gaussian_3d(X, Y, Z, mean=point, sigma=sigma)

# # Step 3: Visualize 3D Gaussian blobs
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Use scatter for better clarity; contour plots can also be used for a grid-based view
# ax.scatter(
#     X.flatten(), Y.flatten(), Z.flatten(), 
#     c=gaussian_splats.flatten(), 
#     cmap='viridis', 
#     alpha=0.5
# )

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("3D Gaussian Splatting (Splat Accumulation)")

# plt.show()

# ____________________________________________________________________________________


# Step 1: Generate random 3D points
num_points = 100
points_3d = np.random.randn(num_points, 3)  # Random points (x, y, z)
colors = np.random.rand(num_points, 3)     # Random RGB colors
weights = np.random.rand(num_points)       # Random weights (0 to 1)

# Enable interactive mode
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for Gaussian points
for i in range(num_points):
    ax.scatter(
        points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 
        color=colors[i], 
        s=weights[i] * 200, 
        alpha=weights[i]
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Interactive 3D Gaussian Splatting")

# Enable interactive features
plt.ion()
plt.show()
# time.sleep(1000.0)