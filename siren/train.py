# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import DataLoader
import tqdm

from core import GradientUtils, ImageSiren, PixelDataset


# Add shot noise
def add_shot_noise(image, scale=255):
    """
    Adds shot noise to the image.

    Parameters:
    - image: numpy array, input image (normalized between [0, 1]).
    - scale: int, scale factor to simulate photon counts (default 255).

    Returns:
    - noisy_image: numpy array, image with shot noise.
    """
    # Scale the image to photon counts
    scaled_image = image * scale
    # Apply Poisson noise
    noisy_image = np.random.poisson(scaled_image)
    # Scale back to the range [0, 1]
    noisy_image = noisy_image / scale
    return np.clip(noisy_image, 0, 1)

# ---------------------------------------------
# Image Loading and Preprocessing
# ---------------------------------------------

# Load the input image
img_ = plt.imread("dog.png")  # Reads the image (assumes grayscale or single-channel)
img = img_.copy()
# # Define downsampling factor for reducing image resolution
# downsampling_factor = 4

# # Normalize the image to the range [-1, 1]
# img = 2 * (img_ - 0.5)

# # Perform downsampling by keeping every 4th pixel in height and width
# img = img[::downsampling_factor, ::downsampling_factor]

# # Store the size of the downsampled image (assuming square images)
size = img.shape[0]

# Create the dataset from the downsampled image
noisy_img = add_shot_noise(img, scale=255)

dataset = PixelDataset(clean_img=img, noisy_img=noisy_img)

# ---------------------------------------------
# Model Training Parameters
# ---------------------------------------------

# Number of training epochs
n_epochs = 1000

# Batch size (equal to the number of pixels in the image)
batch_size = int(size ** 2)

# Frequency of logging and saving intermediate results
logging_freq = 20

# Model configuration: choose between "siren" or "mlp_relu"
model_name = "siren"

# Neural network parameters
hidden_features = 256  # Number of hidden neurons per layer
hidden_layers = 3  # Number of hidden layers

# Training target: "intensity", "grad" (gradient), or "laplace"
target = "laplace"

# ---------------------------------------------
# Model Definition
# ---------------------------------------------

# Create the model based on the specified architecture
if model_name == "siren":
    # SIREN network with sinusoidal activations
    model = ImageSiren(
        hidden_features,
        hidden_layers=hidden_layers,
        hidden_omega=30,  # Frequency factor for SIREN
    )
    model.to("cuda:0")
elif model_name == "mlp_relu":
    # Standard MLP with ReLU activations
    layers = [Linear(2, hidden_features), ReLU()]  # First layer with 2 input features

    # Add hidden layers
    for _ in range(hidden_layers):
        layers.append(Linear(hidden_features, hidden_features))
        layers.append(ReLU())

    # Final output layer
    layers.append(Linear(hidden_features, 1))

    # Create the model from the defined layers
    model = Sequential(*layers)

    # Initialize weights using Xavier initialization for better convergence
    for module in model.modules():
        if not isinstance(module, Linear):
            continue
        torch.nn.init.xavier_normal_(module.weight)
else:
    raise ValueError("Unsupported model")

# ---------------------------------------------
# Training Setup
# ---------------------------------------------

# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=batch_size)

# Define the optimizer (Adam with learning rate 1e-4)
optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

# ---------------------------------------------
# Training Loop
# ---------------------------------------------
for e in range(n_epochs):
    losses = []
    for d_batch in tqdm.tqdm(dataloader):
        # Input is noisy coordinates
        x_batch = d_batch["coords"].to(torch.float32).to("cuda:0")
        x_batch.requires_grad = True
        
        # Noisy input intensities
        y_noisy_batch = d_batch["noisy_intensity"].to(torch.float32)[:, None].to("cuda:0")
        
        # Ground truth clean intensities
        y_clean_batch = d_batch["clean_intensity"].to(torch.float32)[:, None].to("cuda:0")

        # Model predicts clean image
        y_pred_batch = model(x_batch)

        # Loss between predicted clean and ground truth clean
        loss = ((y_clean_batch - y_pred_batch) ** 2).mean()

        losses.append(loss.item())

        # Backpropagation and optimization
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(f"Epoch {e}: Loss = {np.mean(losses)}")

    # Visualization of denoising (optional, save at intervals)
    if e % logging_freq == 0:
        pred_img = np.zeros_like(img)
        for d_batch in tqdm.tqdm(dataloader):
            coords = d_batch["coords"].to(torch.float32).to("cuda:0")
            coords_abs = d_batch["coords_abs"].numpy()

            pred = model(coords).detach().cpu().numpy().squeeze()
            pred_img[coords_abs[:, 0], coords_abs[:, 1]] = pred

        plt.imshow(pred_img, cmap="gray")
        plt.title(f"Iteration {e}: Denoised Image")
        plt.savefig(f"visualization/denoised_{e}.png")
