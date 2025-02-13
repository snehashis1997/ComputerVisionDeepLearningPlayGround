import numpy as np
import cv2

def CullGaussian(p, V):
    """
    Frustum culling to remove Gaussians outside the camera's view.
    """
    # Placeholder for frustum culling logic
    return True  # Assume all Gaussians are visible

def ScreenspaceGaussians(M, S, V):
    """
    Transform Gaussian means and covariances from world space to screen space.
    """
    # Placeholder for transformation logic
    return M, S  # Assume no transformation for simplicity

def CreateTiles(w, h, tile_size=16):
    """
    Create tiles for the image.
    """
    num_tiles_x = (w + tile_size - 1) // tile_size
    num_tiles_y = (h + tile_size - 1) // tile_size
    return [(x, y) for x in range(num_tiles_x) for y in range(num_tiles_y)]

def DuplicateWithKeys(M_prime, T):
    """
    Duplicate Gaussian means with tile keys for sorting.
    """
    L = []
    K = []
    for idx, m in enumerate(M_prime):
        for tile in T:
            L.append((idx, m))
            K.append(tile)
    return L, K

def SortByKeys(K, L):
    """
    Sort Gaussians by tile keys.
    """
    # Convert K to a structured array for sorting
    K_array = np.array(K, dtype=[('x', int), ('y', int)])
    L_array = np.array(L, dtype=[('idx', int), ('m', float, (3,))])
    
    # Sort by tile keys
    sorted_indices = np.argsort(K_array, order=('x', 'y'))
    K_sorted = K_array[sorted_indices]
    L_sorted = L_array[sorted_indices]
    
    return K_sorted, L_sorted

def IdentifyTileRanges(K_sorted):
    """
    Identify ranges of Gaussians for each tile.
    """
    R = {}
    current_tile = None
    start = 0
    for idx, tile in enumerate(K_sorted):
        if tuple(tile) != current_tile:
            if current_tile is not None:
                R[current_tile] = (start, idx - 1)
            current_tile = tuple(tile)
            start = idx
    if current_tile is not None:
        R[current_tile] = (start, len(K_sorted) - 1)
    return R

def BlendInOrder(i, L_sorted, r, K_sorted, M_prime, S_prime, C, A):
    """
    Blend Gaussians in the correct order for a pixel.
    """
    color = np.zeros(3)  # Initialize pixel color
    for idx in range(r[0], r[1] + 1):
        gaussian_idx = L_sorted[idx]['idx']
        # Placeholder for blending logic
        color += C[gaussian_idx] * A[gaussian_idx]  # Simple blending
    return color

def Rasterize(w, h, M, S, C, A, V):
    """
    Rasterize 3D Gaussians into an image.
    """
    # Step 1: Frustum culling
    visible_gaussians = [p for p in range(len(M)) if CullGaussian(p, V)]
    M = [M[i] for i in visible_gaussians]
    S = [S[i] for i in visible_gaussians]
    C = [C[i] for i in visible_gaussians]
    A = [A[i] for i in visible_gaussians]

    # Step 2: Transform to screen space
    M_prime, S_prime = ScreenspaceGaussians(M, S, V)

    # Step 3: Create tiles
    T = CreateTiles(w, h)

    # Step 4: Duplicate with keys and sort
    L, K = DuplicateWithKeys(M_prime, T)
    K_sorted, L_sorted = SortByKeys(K, L)

    # Step 5: Identify tile ranges
    R = IdentifyTileRanges(K_sorted)

    # Step 6: Initialize canvas
    I = np.zeros((h, w, 3))  # RGB image

    # Step 7: Rasterize each tile
    for tile in T:
        tile_x, tile_y = tile
        for i in range(tile_x * 16, min((tile_x + 1) * 16, w)):
            for j in range(tile_y * 16, min((tile_y + 1) * 16, h)):
                r = GetTileRange(R, tile)
                I[j, i] = BlendInOrder((i, j), L_sorted, r, K_sorted, M_prime, S_prime, C, A)

    return I

def GetTileRange(R, tile):
    """
    Get the range of Gaussians for a specific tile.
    """
    return R[tile]

# Example usage
w, h = 800, 600  # Image dimensions
M = [np.array([0, 0, 0]), np.array([1, 1, 1])]  # Gaussian means
S = [np.eye(3), np.eye(3)]  # Gaussian covariances
C = [np.array([1, 0, 0]), np.array([0, 1, 0])]  # Gaussian colors
A = [0.5, 0.5]  # Gaussian opacities
V = {}  # View configuration (placeholder)

image = Rasterize(w, h, M, S, C, A, V)
image = np.array(image*255, dtype=np.uint8)
print(np.unique(image))
cv2.imwrite("1.jpg", image)