import torch
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Initialize 3D points
num_points = 1000
points_3d = torch.randn((num_points, 3))  # Random 3D points (x, y, z)
colors = torch.rand((num_points, 3))     # Random colors for each point (RGB)

# Step 2: Define Gaussian splats
def gaussian_2d(x, y, mean, sigma):
    """Generate a 2D Gaussian."""
    return torch.exp(-((x - mean[0])**2 + (y - mean[1])**2) / (2 * sigma**2))

# Step 3: Project 3D points to 2D (simple perspective projection)
def project_points(points, focal_length=1.0, img_size=512):
    """Project 3D points to 2D using perspective projection."""
    z = points[:, 2].clamp(min=1e-3)  # Avoid division by zero
    x_2d = (points[:, 0] / z * focal_length + 0.5) * img_size
    y_2d = (points[:, 1] / z * focal_length + 0.5) * img_size
    return torch.stack((x_2d, y_2d), dim=-1)

# Project points to 2D
image_size = 512
projected_points = project_points(points_3d, focal_length=1.0, img_size=image_size)

# Step 4: Accumulate Gaussian splats
canvas = torch.zeros((image_size, image_size, 3))  # Blank canvas
sigma = 5.0  # Gaussian spread

for i in range(num_points):
    x, y = projected_points[i]
    x, y = int(x.item()), int(y.item())
    if 0 <= x < image_size and 0 <= y < image_size:
        # Create a Gaussian splat
        grid_x, grid_y = torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing="ij"
        )
        gaussian = gaussian_2d(grid_x, grid_y, mean=(x, y), sigma=sigma)
        gaussian = gaussian.unsqueeze(-1)  # Add color channel dimension
        canvas += gaussian * colors[i]  # Accumulate color contributions

# Step 5: Normalize and visualize
canvas = canvas / canvas.max()  # Normalize to [0, 1] for visualization
plt.imshow(canvas.numpy())
plt.axis("off")
plt.show()
