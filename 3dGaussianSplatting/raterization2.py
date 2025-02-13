import numpy as np

def frustum_cull(gaussians, view_config):
    # Implement basic frustum culling
    visible_gaussians = []
    for gaussian in gaussians:
        # Simple culling logic - replace with proper frustum check
        if np.all(np.abs(gaussian['mean']) < view_config['max_range']):
            visible_gaussians.append(gaussian)
    return visible_gaussians

def transform_to_screenspace(gaussians, view_matrix):
    # Transform Gaussians to screen space
    screenspace_gaussians = []
    for gaussian in gaussians:
        # Apply view matrix transformation
        transformed_mean = view_matrix @ gaussian['mean']
        transformed_cov = view_matrix @ gaussian['cov'] @ view_matrix.T
        screenspace_gaussians.append({
            'mean': transformed_mean,
            'cov': transformed_cov,
            'color': gaussian['color'],
            'opacity': gaussian['opacity']
        })
    return screenspace_gaussians

def create_tiles(width, height, tile_size=16):
    # Create grid of tiles
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tiles.append({
                'x': x, 
                'y': y, 
                'width': min(tile_size, width - x),
                'height': min(tile_size, height - y)
            })
    return tiles

def compute_tile_keys(gaussians, tiles):
    # Compute tile keys for each Gaussian
    tile_keys = []
    for i, gaussian in enumerate(gaussians):
        for j, tile in enumerate(tiles):
            # Check if Gaussian intersects with tile (simplified)
            if (gaussian['mean'][0] >= tile['x'] and 
                gaussian['mean'][0] < tile['x'] + tile['width'] and
                gaussian['mean'][1] >= tile['y'] and 
                gaussian['mean'][1] < tile['y'] + tile['height']):
                tile_keys.append((j, i))
                break
    return tile_keys

def rasterize(width, height, gaussians, view_config):
    # Main rasterization function
    # Cull and transform Gaussians
    visible_gaussians = frustum_cull(gaussians, view_config)
    screenspace_gaussians = transform_to_screenspace(visible_gaussians, view_config['view_matrix'])
    
    # Create tiles and compute keys
    tiles = create_tiles(width, height)
    tile_keys = compute_tile_keys(screenspace_gaussians, tiles)
    
    # Sort Gaussians by tile keys
    sorted_indices = np.argsort(tile_keys)
    sorted_gaussians = [screenspace_gaussians[i] for i in sorted_indices]
    
    # Initialize output image
    image = np.zeros((height, width, 4), dtype=np.float32)
    
    # Blend Gaussians
    for gaussian in sorted_gaussians:
        x, y = int(gaussian['mean'][0]), int(gaussian['mean'][1])
        if 0 <= x < width and 0 <= y < height:
            # Simplified blending (replace with proper Gaussian splatting)
            image[y, x] = gaussian['color'] * gaussian['opacity']
    
    return image

# Example usage
def main():
    # Example Gaussian data
    gaussians = [
        {
            'mean': np.array([100, 100, 0]),
            'cov': np.eye(3),
            'color': np.array([1, 0, 0]),  # Red
            'opacity': 0.5
        },
        # Add more Gaussians...
    ]
    
    view_config = {
        'view_matrix': np.eye(4),
        'max_range': 1000
    }
    
    result = rasterize(400, 300, gaussians, view_config)
    print(np.unique(result))
    # Visualize or save result

main()