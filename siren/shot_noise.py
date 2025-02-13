import numpy as np
import matplotlib.pyplot as plt

# Load your image (assuming it is grayscale and normalized between 0 and 1)
image = plt.imread("dog.png")  # Load image
if image.max() > 1:  # Normalize if necessary
    image = image / 255.0

def add_shot_noise_more(image, scale=100):  # Reduce scale for more noise
    """
    Adds more shot noise to the image.

    Parameters:
    - image: numpy array, input image (normalized between [0, 1]).
    - scale: int, scale factor to simulate photon counts (lower scale increases noise).

    Returns:
    - noisy_image: numpy array, image with more shot noise.
    """
    # Scale the image to photon counts
    scaled_image = image * scale
    # Apply Poisson noise
    noisy_image = np.random.poisson(scaled_image)
    # Scale back to the range [0, 1]
    noisy_image = noisy_image / scale
    return np.clip(noisy_image, 0, 1)

# Lower `scale` will make photon counts sparse, adding more noise
noisy_image = add_shot_noise_more(image, scale=1)  # Experiment with smaller scale values


# Visualize the original and noisy images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Shot Noise Image")
plt.imshow(noisy_image, cmap="gray")
plt.axis("off")

plt.show()

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# def add_shot_noise(image):
#     # Ensure the image is in floating-point format
#     image = image.astype(np.float32) / 255.0

#     # Scale the image to represent photon counts
#     photon_count = image * 1000  # Adjust scaling factor as needed

#     # Add Poisson noise
#     noisy_image = np.random.poisson(photon_count).astype(np.float32)

#     # Normalize back to the original range (0-1)
#     noisy_image = noisy_image / np.max(noisy_image)

#     # Convert back to 8-bit format (0-255)
#     noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

#     return noisy_image

# # Load an image
# image = cv2.imread('dog.png', cv2.IMREAD_GRAYSCALE)

# # Add shot noise
# noisy_image = add_shot_noise(image)

# # Display the results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title('Image with Shot Noise')
# plt.imshow(noisy_image, cmap='gray')

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def single_photon_camera_simulator(image, exposure_time=0.1, dark_count_rate=0.01, dead_time=0.0, scale=255):
#     """
#     Simulates a single-photon camera for an input image.

#     Parameters:
#     - image: numpy array, input image (normalized between [0, 1]).
#     - exposure_time: float, time interval during which photons are counted (in seconds).
#     - dark_count_rate: float, rate of dark noise (per pixel per second).
#     - dead_time: float, dead time after each photon detection (in seconds).
#     - scale: int, scale factor to simulate photon counts (default 255).

#     Returns:
#     - photon_count_image: numpy array, simulated photon count image.
#     """
#     # Ensure image is normalized
#     if image.max() > 1:
#         image = image / 255.0

#     # Scale image to simulate photon arrival rate (e.g., photon flux per pixel)
#     photon_rate = image * scale

#     # Simulate photon arrival using Poisson distribution
#     photon_counts = np.random.poisson(photon_rate * exposure_time)

#     # Simulate dark counts (independent noise)
#     dark_counts = np.random.poisson(dark_count_rate * exposure_time, size=image.shape)

#     # Combine photon counts with dark counts
#     total_counts = photon_counts + dark_counts

#     # Simulate dead time (optional, simple implementation by clipping max counts)
#     if dead_time > 0:
#         max_counts = int(exposure_time / dead_time)
#         total_counts = np.clip(total_counts, 0, max_counts)

#     # Normalize photon count image to [0, 1] for visualization
#     photon_count_image = total_counts / total_counts.max()

#     return photon_count_image

# # Load an image (grayscale and normalized)
# image = plt.imread("dog.png")
# if image.ndim == 3:  # Convert RGB to grayscale if necessary
#     image = np.mean(image, axis=-1)
# if image.max() > 1:  # Normalize if necessary
#     image = image / 255.0

# # Simulate single-photon camera
# simulated_image = single_photon_camera_simulator(image, exposure_time=0.1, dark_count_rate=0.01, dead_time=0.0, scale=255)

# # Visualize the original and simulated images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap="gray")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.title("Simulated Single-Photon Image")
# plt.imshow(simulated_image, cmap="gray")
# plt.axis("off")

# plt.show()
