import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define 2D keypoints for the same human from two camera views
keypoints_cam1 = np.array([
    [870, 347], [873, 341], [865, 341], [881, 347], [857, 347],
    [894, 376], [841, 376], [910, 408], [814, 403], [902, 430],
    [827, 430], [881, 451], [849, 454], [883, 518], [851, 518],
    [881, 579], [857, 576], [867, 376]
], dtype=float)

keypoints_cam2 = np.array([
    [903, 295], [901, 290], [898, 290], [887, 292], [884, 292],
    [884, 326], [872, 326], [878, 371], [850, 357], [895, 388],
    [872, 393], [878, 410], [864, 410], [872, 480], [858, 483],
    [864, 551], [853, 556], [878, 326]
], dtype=float)

# Camera projection matrices
P1 = np.array([
    [1400, 0, 640, 0],
    [0, 1400, 360, 0],
    [0, 0, 1, 0]
], dtype=float)

P2 = np.array([
    [1400, 0, 640, -100],
    [0, 1400, 360, 0],
    [0, 0, 1, 0]
], dtype=float)

# Triangulate 3D points
keypoints_cam1 = keypoints_cam1.T  # Shape (2, N)
keypoints_cam2 = keypoints_cam2.T  # Shape (2, N)

points_4d = cv2.triangulatePoints(P1, P2, keypoints_cam1, keypoints_cam2)

# Convert from homogeneous to 3D coordinates
points_3d = points_4d[:3] / points_4d[3]

# Transpose to get Nx3 shape
points_3d = points_3d.T

# Define line pairs and colors
l_pair = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
          (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
          (17, 11), (17, 12),  # Body
          (11, 13), (12, 14), (13, 15), (14, 16)]
line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

# Normalize colors to 0-1 range for Matplotlib
line_color = [(r / 255, g / 255, b / 255) for r, g, b in line_color]

# Plot 3D points and lines
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o', s=30, label="Keypoints")
# print(points_3d)

# Draw lines
for i, (start, end) in enumerate(l_pair):
    x_coords = [points_3d[start, 0], points_3d[end, 0]]
    y_coords = [points_3d[start, 1], points_3d[end, 1]]
    z_coords = [points_3d[start, 2], points_3d[end, 2]]
    ax.plot(x_coords, y_coords, z_coords, color=line_color[i], linewidth=2)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add legend and show plot
plt.legend()
plt.show()
