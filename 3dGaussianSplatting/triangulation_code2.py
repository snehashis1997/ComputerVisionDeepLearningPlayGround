# triangulations uncalibrated stereo code ----------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Given 2D keypoints in two views
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

# 1. Estimate the Fundamental Matrix
F, mask = cv2.findFundamentalMat(keypoints_cam1, keypoints_cam2, method=cv2.FM_8POINT)

# Keep only inliers
inlier_keypoints_cam1 = keypoints_cam1[mask.ravel() == 1]
inlier_keypoints_cam2 = keypoints_cam2[mask.ravel() == 1]

# 2. Recover Projection Matrices
# First camera's projection matrix (canonical form)
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

# Compute the epipole (right null vector of F)
U, S, Vt = np.linalg.svd(F)
e = Vt[-1]
e /= e[2]  # Normalize to make it homogeneous

# Construct the skew-symmetric matrix of e
e_skew = np.array([
    [0, -e[2], e[1]],
    [e[2], 0, -e[0]],
    [-e[1], e[0], 0]
])

# Compute the second projection matrix
P2 = np.hstack((e_skew @ F.T, e.reshape(-1, 1)))

# 3. Triangulate Points
keypoints_cam1_h = np.hstack((inlier_keypoints_cam1, np.ones((inlier_keypoints_cam1.shape[0], 1)))).T
keypoints_cam2_h = np.hstack((inlier_keypoints_cam2, np.ones((inlier_keypoints_cam2.shape[0], 1)))).T

# Triangulate points
points_4d = cv2.triangulatePoints(P1, P2, keypoints_cam1_h[:2], keypoints_cam2_h[:2])

# Convert from homogeneous to 3D
points_3d = points_4d[:3] / points_4d[3]

print(points_3d)

# 4. Visualize
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_3d[0], points_3d[1], points_3d[2], c='r', marker='o')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.title('Reconstructed 3D Points')
# plt.show()

# Define line pairs and colors
l_pair = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
          (5, 6), (5, 7), (7, 9), (6, 8), 
          (8, 10),(17, 11), (17, 12), (11, 13), 
          (12, 14), (13, 15), (14, 16)]
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



# triangulations uncalibrated stereo using code ----------------------

# import numpy as np
# import cv2

# def uncalibrated_stereo(keypoints_cam1, keypoints_cam2):
#     """
#     Performs uncalibrated stereo reconstruction.

#     Args:
#         keypoints_cam1: Nx2 array of keypoints in camera 1.
#         keypoints_cam2: Nx2 array of corresponding keypoints in camera 2.

#     Returns:
#         points_3d: Nx3 array of triangulated 3D points.
#         P1: 3x4 projection matrix for camera 1.
#         P2: 3x4 projection matrix for camera 2.  Returns None if fundamental matrix calculation fails.
#     """

#     # 1. Estimate the Fundamental Matrix (F)
#     F, mask = cv2.findFundamentalMat(keypoints_cam1, keypoints_cam2, cv2.FM_RANSAC, 3)

#     if F is None:
#         print("Fundamental matrix calculation failed.")
#         return None, None, None

#     # Keep only inlier points based on the mask
#     keypoints_cam1 = keypoints_cam1[mask.ravel() == 1]
#     keypoints_cam2 = keypoints_cam2[mask.ravel() == 1]

#     # 2. Recover Camera Matrices (P1 and P2) - simplified approach for uncalibrated case
#     #   We'll use a canonical form for P1 and derive P2 from F.  This doesn't
#     #   recover the true projective geometry, but it gives a 3D reconstruction
#     #   up to a projective transformation.

#     P1 = np.array([[1, 0, 0, 0],
#                    [0, 1, 0, 0],
#                    [0, 0, 1, 0]])

#     # Deriving P2 from F (up to a projective transformation)
#     # This is a simplified approach, a more robust method would involve RQ decomposition
#     # of a related matrix.
    
#     #Skew-symmetric matrix corresponding to the epipole (0,0,0,1) in the first camera
#     E = np.array([[0, -F[2,1], F[1,1]],
#                   [F[2,1], 0, -F[0,1]],
#                   [-F[1,1], F[0,1], 0]])

#     P2 = np.concatenate((F.T @ E, F[:,2].reshape(3,1)), axis=1)

#     # 3. Triangulate Points
#     keypoints_cam1_homogeneous = cv2.convertPointsToHomogeneous(keypoints_cam1)[:, 0, :].T
#     keypoints_cam2_homogeneous = cv2.convertPointsToHomogeneous(keypoints_cam2)[:, 0, :].T

#     points_4d = cv2.triangulatePoints(P1, P2, keypoints_cam1_homogeneous, keypoints_cam2_homogeneous)

#     # Convert from homogeneous coordinates to 3D
#     points_3d = points_4d[:3, :].T / points_4d[3, :]

#     return points_3d, P1, P2


# # Example usage (your provided keypoints):
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


# points_3d, P1, P2 = uncalibrated_stereo(keypoints_cam1, keypoints_cam2)
# print(P1.shape, P2.shape)

# if points_3d is not None:
#     print("3D Points:\n", points_3d)
#     print("Projection Matrix P1:\n", P1)
#     print("Projection Matrix P2:\n", P2)

#     # You can now visualize the 3D points (e.g., using matplotlib)
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
#     plt.show()

# triangulations uncalibrated stereo using code ----------------------

# import numpy as np
# import cv2

# # Provided keypoints
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

# # Step 1: Find Corresponding Points (already provided)

# # Step 2: Estimate the Fundamental Matrix
# F, mask = cv2.findFundamentalMat(keypoints_cam1, keypoints_cam2, cv2.FM_RANSAC)

# # Step 3: Recover Camera Matrices
# # Assuming the first camera is at the origin (P1 = [I | 0])
# P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

# # Compute the Essential Matrix (if intrinsic parameters are known)
# # For uncalibrated stereo, we directly compute P2 from F
# # Using the method described in Hartley & Zisserman's book
# e2 = cv2.computeCorrespondEpilines(keypoints_cam1.reshape(-1, 1, 2), 1, F)
# e2 = e2.reshape(-1, 3)

# # Compute the second camera matrix P2
# _, _, _, _, _, _, P2 = cv2.recoverPose(F, keypoints_cam1, keypoints_cam2)

# # Step 4: Triangulate Points
# points_4d_hom = cv2.triangulatePoints(P1, P2, keypoints_cam1.T, keypoints_cam2.T)
# points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Convert to non-homogeneous coordinates

# # Print the 3D points
# print("3D Points:")
# print(points_3d.T)

# import numpy as np
# import cv2

# # Provided keypoints
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

# # Step 1: Estimate the Fundamental Matrix
# F, mask = cv2.findFundamentalMat(keypoints_cam1, keypoints_cam2, cv2.FM_RANSAC)

# # Filter inlier points based on the mask
# keypoints_cam1_inliers = keypoints_cam1[mask.ravel() == 1]
# keypoints_cam2_inliers = keypoints_cam2[mask.ravel() == 1]

# # Step 2: Compute the Essential Matrix (if intrinsic matrix K is known)
# # For uncalibrated stereo, skip this step.
# # Example (if K is known):
# # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# # E = K.T @ F @ K

# # Step 3: Recover Pose from F or E
# # Since K is not provided, use F directly (less reliable but works for uncalibrated stereo)
# _, R, t, mask_pose = cv2.recoverPose(F, keypoints_cam1_inliers, keypoints_cam2_inliers)
# P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
# P2 = np.hstack((R, t))

# # Step 4: Triangulate Points
# points_4d_hom = cv2.triangulatePoints(P1, P2, keypoints_cam1_inliers.T, keypoints_cam2_inliers.T)
# points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Convert to non-homogeneous coordinates

# # Print the 3D points
# # print("3D Points:")
# # print(points_3d.T)

# # # 4. Visualize
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_3d[0], points_3d[1], points_3d[2], c='r', marker='o')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.title('Reconstructed 3D Points')
# plt.show()