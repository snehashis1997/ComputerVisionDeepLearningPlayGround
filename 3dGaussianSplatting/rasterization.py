import numpy as np
import cv2

def CullGaussian(M, V):
    """
    Perform frustum culling to exclude Gaussians outside the visible frustum.

    Args:
        M (np.array): Gaussian means in world space.
        V (np.array): View configuration (projection matrix).

    Returns:
        np.array: Filtered Gaussian means within the visible frustum.
    """
    culled_means = []
    for mean in M:
        clip_space = V @ np.append(mean, 1.0)  # Transform to clip space
        if all(-clip_space[3] <= clip_space[i] <= clip_space[3] for i in range(3)):
            culled_means.append(mean)
    return np.array(culled_means)

def ScreenspaceGaussians(M, S, V):
    """
    Transform Gaussians to screen space.

    Args:
        M (np.array): Gaussian means in world space.
        S (np.array): Gaussian covariances in world space.
        V (np.array): View configuration (projection matrix).

    Returns:
        tuple: Transformed means and covariances in screen space.
    """
    M_prime = []
    S_prime = []
    for mean, cov in zip(M, S):
        clip_space = V @ np.append(mean, 1.0)
        ndc_space = clip_space[:3] / clip_space[3]  # Normalize to NDC
        screen_space = np.array([
            0.5 * (ndc_space[0] + 1) * w,  # Convert to screen space
            0.5 * (ndc_space[1] + 1) * h,
            ndc_space[2]
        ])
        M_prime.append(screen_space)
        S_prime.append(cov)  # Placeholder: transform covariances if needed
    return np.array(M_prime), np.array(S_prime)

def CreateTiles(w, h, tile_size=16):
    """
    Divide the screen into tiles of size tile_size x tile_size.

    Args:
        w (int): Width of the screen.
        h (int): Height of the screen.
        tile_size (int): Size of each tile.

    Returns:
        list: List of tile bounding boxes.
    """
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tiles.append((x, y, x + tile_size, y + tile_size))
    return tiles

def DuplicateWithKeys(M_prime, tiles):
    """
    Duplicate Gaussians for each tile and generate sort keys based on distance.

    Args:
        M_prime (np.array): Gaussian means in screen space.
        tiles (list): List of tile bounding boxes.

    Returns:
        tuple: List of duplicated Gaussians and corresponding keys.
    """
    L = []
    K = []
    for i, mean in enumerate(M_prime):
        for tile_idx, (x_min, y_min, x_max, y_max) in enumerate(tiles):
            if x_min <= mean[0] < x_max and y_min <= mean[1] < y_max:
                L.append(np.append(mean, tile_idx))  # Append tile index to Gaussian
                K.append(mean[2])  # Depth as sort key
    return np.array(L), np.array(K)

def IdentifyTileRanges(tiles, K, L):
    """
    Identify the range of indices for each tile based on the sorted keys.

    Args:
        tiles (list): List of all tiles.
        K (list): Sorted keys corresponding to Gaussians.
        L (list): Sorted Gaussians or related data.

    Returns:
        dict: Dictionary mapping tile indices to ranges of Gaussian indices.
    """
    tile_ranges = {}
    for idx, tile in enumerate(tiles):
        start_idx = None
        end_idx = None
        for i in range(len(L)):
            if L[i][3] == idx:  # Assuming L[i][3] contains the tile index
                if start_idx is None:
                    start_idx = i
                end_idx = i
        if start_idx is not None:
            tile_ranges[idx] = (start_idx, end_idx + 1)  # Inclusive range
        else:
            tile_ranges[idx] = None  # No Gaussians in this tile
    return tile_ranges

def BlendInOrder(pixel, L, r, K, M_prime, S_prime, C, A, I):
    """
    Blend Gaussians into the canvas in order of depth.

    Args:
        pixel (tuple): (x, y) coordinates of the pixel.
        L (list): Sorted list of Gaussians.
        r (tuple): Range of indices for the tile.
        K (list): Sorted keys corresponding to Gaussians.
        M_prime (list): Gaussian means in screen space.
        S_prime (list): Gaussian covariances in screen space.
        C (list): Gaussian colors.
        A (list): Gaussian opacities.
        I (np.array): Canvas to update.

    Returns:
        np.array: Updated canvas with blended values.
    """
    if r is None or len(r) != 2:
        return I

    start_idx, end_idx = r
    x, y = pixel
    for i in range(start_idx, end_idx):
        gaussian_color = C[i]
        gaussian_opacity = A[i]
        distance = np.linalg.norm(np.array([x, y]) - M_prime[i][:2])

        existing_color = I[y, x, :3]  # RGB
        existing_opacity = I[y, x, 3]  # Alpha

        blended_color = gaussian_opacity * gaussian_color[:3] + (1 - gaussian_opacity) * existing_color
        blended_opacity = gaussian_opacity + (1 - gaussian_opacity) * existing_opacity

        I[y, x, :3] = blended_color
        I[y, x, 3] = blended_opacity

    return I

def Rasterize(w, h, M, S, C, A, V):
    """
    Perform rasterization of 3D Gaussians.

    Args:
        w (int): Width of the image.
        h (int): Height of the image.
        M (np.array): Gaussian means in world space.
        S (np.array): Gaussian covariances in world space.
        C (np.array): Gaussian colors.
        A (np.array): Gaussian opacities.
        V (np.array): View configuration (projection matrix).

    Returns:
        np.array: Rasterized image.
    """
    M = CullGaussian(M, V)
    M_prime, S_prime = ScreenspaceGaussians(M, S, V)
    tiles = CreateTiles(w, h)
    L, K = DuplicateWithKeys(M_prime, tiles)
    sorted_indices = np.argsort(K)
    L = L[sorted_indices]
    K = K[sorted_indices]
    R = IdentifyTileRanges(tiles, K, L)

    I = np.zeros((h, w, 4))  # Initialize canvas with RGBA

    for tile_idx, (x_min, y_min, x_max, y_max) in enumerate(tiles):
        if R[tile_idx] is not None:
            start_idx, end_idx = R[tile_idx]
            for y in range(y_min, min(y_max, h)):
                for x in range(x_min, min(x_max, w)):
                    I = BlendInOrder((x, y), L, (start_idx, end_idx), K, M_prime, S_prime, C, A, I)

    return I

# Example usage
w, h = 800, 600
M = np.random.rand(10, 3) * 10  # Random Gaussian means
S = np.random.rand(10, 3, 3)  # Random Gaussian covariances
C = np.random.rand(10, 3)  # Random colors
A = np.random.rand(10)  # Random opacities
# Simple perspective projection matrix (for testing)
fov = 45  # Field of view in degrees
aspect_ratio = w / h
near = 0.1
far = 100.0
top = near * np.tan(np.radians(fov / 2))
right = top * aspect_ratio

V = np.array([
    [near / right, 0, 0, 0],
    [0, near / top, 0, 0],
    [0, 0, -(far + near) / (far - near), -1],
    [0, 0, -2 * far * near / (far - near), 0]
])

result_image = Rasterize(w, h, M, S, C, A, V)
print(np.unique(result_image))
cv2.imwrite("1.jpg", result_image)

# import numpy as np

# # Placeholder functions - replace with actual implementations
# def CullGaussian(p, V):
#     """Performs frustum culling. Placeholder."""
#     # In a real implementation, this would check if Gaussians are within the camera's view.
#     return True  # Or False, based on the actual culling logic

# def ScreenspaceGaussians(M, S, V):
#     """Transforms Gaussians from world space to screen space. Placeholder."""
#     # This would involve matrix transformations using the view matrix V.
#     return M, S  # Return transformed M and S

# def CreateTiles(w, h):
#     """Creates a tiling structure for the image. Placeholder."""
#     # This would divide the image into tiles (e.g., 8x8 or 16x16 pixels).
#     return [(x, y) for x in range(0, w, 8) for y in range(0, h, 8)] # Example: 8x8 tiles

# def DuplicateWithKeys(M_prime, T):
#     """Duplicates data with keys. Placeholder."""
#     # This would associate each Gaussian with the tiles it overlaps.
#     K = []
#     L = []
#     for i, gaussian in enumerate(M_prime): # Assuming M_prime is a list of Gaussian centers
#         for tile in T:
#             K.append(tile) # Simplified: Every gaussian in every tile (not efficient but illustrative)
#             L.append(gaussian) # Keep a list of the gaussian data
#     return L, K

# def SortByKeys(K, L):
#     combined = sorted(zip(K, L), key=lambda item: tuple(item[0])) # Convert array to tuple for comparison
#     K_sorted, L_sorted = zip(*combined) if combined else ([], [])
#     return list(K_sorted), list(L_sorted)


# def IdentifyTileRanges(T, K):
#   """Identifies ranges of tiles in the sorted list. Placeholder."""
#   R = {}
#   start_index = 0
#   current_tile = None
#   for i, tile in enumerate(K):
#       if tile != current_tile:
#           if current_tile is not None:
#               R[current_tile] = (start_index, i)  # Store the range
#           current_tile = tile
#           start_index = i
#   if current_tile is not None:  # Add the last tile
#       R[current_tile] = (start_index, len(K))
#   return R


# def GetTileRange(R, t):
#     """Retrieves the range of a tile. Placeholder."""
#     return R.get(t) # Returns None if the tile is not present, handle as needed

# def BlendInOrder(i, L, r, K, M_prime, S_prime, C, A):
#     x, y = i
#     accumulated_color = np.array([0.0, 0.0, 0.0])

#     # print(f"Pixel: ({x}, {y})")
#     if r is not None:
#         for j in range(r[0], r[1]):  # j is the INTEGER index
#             # gaussian_index = K[j]  # Don't use this directly
#             # print("Gaussian Index:", gaussian_index)
#             # mx, my = M_prime[gaussian_index]  # Incorrect: tuple index
#             print(j, len(M_prime), r[1])
#             mx, my = M_prime[j]  # Correct: integer index j
#             # print(f"Gaussian Center: ({mx}, {my})")

#             # Gaussian Influence Calculation (REPLACE WITH YOUR ACTUAL CALCULATION)
#             dx = x - mx
#             dy = y - my
#             influence = np.exp(-(dx**2 + dy**2) / (2 * 10**2))  # Example

#             # print("Influence:", influence)
#             color = np.array(C[j]) * A[j] * influence  # Access C and A with j as well
#             # print("Color:", color)

#             accumulated_color += color

#     return accumulated_color


# def RASTERIZE(w, h, M, S, C, A, V):
#     """Main rasterization function."""

#     # M, S, C, A should be lists or numpy arrays of appropriate sizes
#     # representing the parameters for each Gaussian.
    
#     M_prime, S_prime = ScreenspaceGaussians(M, S, V)

#     T = CreateTiles(w, h)

#     L, K = DuplicateWithKeys(M_prime, T)
#     K_sorted, L_sorted = SortByKeys(K, L)  # Sort by tile key
#     R = IdentifyTileRanges(T, K_sorted)

#     I = np.zeros((h, w, 3))  # Initialize image buffer (3 for RGB)

#     for tile in T:
#         tile_range = GetTileRange(R, tile)
#         if tile_range:  # Check if the tile has any associated Gaussians
#             start, end = tile_range
#             for y in range(tile[1], min(tile[1] + 8, h)): # Iterate over pixels in tile
#                 for x in range(tile[0], min(tile[0] + 8, w)):
#                     i = (y, x) # Pixel coordinates
#                     I[i] = BlendInOrder(i, L_sorted, tile_range, K_sorted, M_prime, S_prime, C, A)

#     return I


# # Example usage (replace with your actual data):
# w, h = 256, 256
# num_gaussians = 100
# M = [np.random.rand(2) * [w, h] for _ in range(num_gaussians)] # Example Gaussian centers
# S = [np.random.rand(2, 2) for _ in range(num_gaussians)] # Example covariance matrices
# C = [np.random.rand(3) for _ in range(num_gaussians)] # Example colors
# A = [np.random.rand() for _ in range(num_gaussians)] # Example opacities
# # V = None # Replace with your actual view matrix

# # Simple perspective projection matrix (for testing)
# fov = 45  # Field of view in degrees
# aspect_ratio = w / h
# near = 0.1
# far = 100.0
# top = near * np.tan(np.radians(fov / 2))
# right = top * aspect_ratio

# V = np.array([
#     [near / right, 0, 0, 0],
#     [0, near / top, 0, 0],
#     [0, 0, -(far + near) / (far - near), -1],
#     [0, 0, -2 * far * near / (far - near), 0]
# ])

# image = RASTERIZE(w, h, M, S, C, A, V)

# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.show()