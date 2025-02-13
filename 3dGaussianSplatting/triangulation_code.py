# import cv2
# import numpy as np

# # Define 2D keypoints for the same human from two camera views
# keypoints_cam1 = np.array([
#     [870, 347], [873, 341], [865, 341], [881, 347], [857, 347],
#     [894, 376], [841, 376], [910, 408], [814, 403], [902, 430],
#     [827, 430], [881, 451], [849, 454], [883, 518], [851, 518],
#     [881, 579], [857, 576], [867, 376]
# ], dtype=float)

# keypoints_cam2 = np.array([
#     [903, 295], [901, 290], [898, 290], [887, 292], [884, 292],
#     [884, 326], [872, 326], [878, 371], [850, 357], [895, 388],
#     [872, 393], [878, 410], [864, 410], [872, 480], [858, 483],
#     [864, 551], [853, 556], [878, 326]
# ], dtype=float)

# # Camera projection matrices (example values)
# # Replace these with actual intrinsic/extrinsic matrices from calibration
# P1 = np.array([
#     [1400, 0, 640, 0],
#     [0, 1400, 360, 0],
#     [0, 0, 1, 0]
# ], dtype=float)

# P2 = np.array([
#     [1400, 0, 640, -100],
#     [0, 1400, 360, 0],
#     [0, 0, 1, 0]
# ], dtype=float)

# # Triangulate 3D points
# keypoints_cam1 = keypoints_cam1.T  # Shape (2, N)
# keypoints_cam2 = keypoints_cam2.T  # Shape (2, N)

# points_4d = cv2.triangulatePoints(P1, P2, keypoints_cam1, keypoints_cam2)

# # Convert from homogeneous to 3D coordinates
# points_3d = points_4d[:3] / points_4d[3]

# # Transpose to get Nx3 shape
# points_3d = points_3d.T

# # Print 3D points
# print("3D Points:")
# print(points_3d)

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

# Camera projection matrices (example values)
# Replace these with actual intrinsic/extrinsic matrices from calibration
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

l_pair = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
          (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
          (17, 11), (17, 12),  # Body
          (11, 13), (12, 14), (13, 15), (14, 16)]

p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
            (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
            (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                (77, 222, 255), (255, 156, 127),
                (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]


# Draw limbs
for i, (start_p, end_p) in enumerate(l_pair):
    if start_p in part_line and end_p in part_line:
        start_xy = part_line[start_p]
        end_xy = part_line[end_p]
        cv2.line(img, start_xy, end_xy, (255,255,255), 1)

        # if i < len(line_color):
            # if opt.tracking:
            #     cv2.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
            # else:
            #     cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
        # else:
            # cv2.line(img, start_xy, end_xy, (255,255,255), 1)


# Plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

# Print 3D points
print("3D Points:")
print(points_3d)
