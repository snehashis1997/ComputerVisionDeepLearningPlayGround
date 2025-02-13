import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset


def paper_init_(weight, is_first=False, omega=1):
    """
    Initialize the weights of the linear layer following the SIREN paper's method.

    Parameters
    ----------
    weight : torch.Tensor
        The weight matrix of the linear layer to be initialized.

    is_first : bool, optional
        If True, indicates this is the first layer in the network, and 
        weights are initialized differently. Default is False.

    omega : float, optional
        Scaling factor for the sine activation. Default is 1.

    Notes
    -----
    - For the first layer, weights are initialized to small random values.
    - For subsequent layers, weights are scaled based on the input dimensionality 
      and the omega parameter to maintain stability in the network.
    """
    in_features = weight.shape[1]  # Number of input features

    with torch.no_grad():
        if is_first:
            # Initialize with smaller bounds for the first layer
            bound = 1 / in_features
        else:
            # Use bounds derived from the SIREN paper for subsequent layers
            bound = np.sqrt(6 / in_features) / omega

        # Uniformly initialize weights within the range [-bound, bound]
        weight.uniform_(-bound, bound)

class SineLayer(nn.Module):
    """
    A Linear layer followed by the sine activation function.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer.

    out_features : int
        Number of output features for the layer.

    bias : bool, optional
        If True (default), a bias term is included in the linear transformation.

    is_first : bool, optional
        If True, this layer is the first layer in the network. 
        This affects the initialization of the weights.

    omega : int, optional
        A hyperparameter that scales the input to the sine activation 
        function. Default is 30.

    custom_init_function_ : None or callable, optional
        If None (default), the layer uses the `paper_init_` function to initialize weights.
        If a callable is provided, it will be used to initialize the weights.

    Attributes
    ----------
    omega : int
        Scaling factor for the sine activation function.

    linear : nn.Linear
        PyTorch's linear layer that performs the linear transformation.

    Notes
    -----
    This layer is part of SIREN (Sinusoidal Representation Networks), 
    where the sine activation function is used to model highly detailed 
    and high-frequency functions effectively.
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=30,
            custom_init_function_=None,
    ):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize weights of the linear layer
        if custom_init_function_ is None:
            # Use default initialization as per the SIREN paper
            paper_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            # Use custom initialization if provided
            custom_init_function_(self.linear.weight)

    def forward(self, x):
        """
        Perform the forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, in_features)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(n_samples, out_features)`, after 
            applying the linear transformation and sine activation function.
        """
        return torch.sin(self.omega * self.linear(x))


class ImageSiren(nn.Module):
    """
    A neural network composed of multiple SineLayers for modeling image-like data.

    Parameters
    ----------
    hidden_features : int
        Number of hidden features in each SineLayer.

    hidden_layers : int, optional
        Number of hidden layers in the network. Default is 1.

    first_omega : float, optional
        Scaling factor (omega) for the sine activation in the first layer. Default is 30.

    hidden_omega : float, optional
        Scaling factor (omega) for the sine activation in the hidden layers. Default is 30.

    custom_init_function_ : None or callable, optional
        If None (default), the `paper_init_` function is used for weight initialization.
        Otherwise, a user-defined callable function can be provided to initialize weights.

    Attributes
    ----------
    net : nn.Sequential
        Sequential collection of `SineLayer`s, followed by a final linear layer.
    """
    def __init__(
            self,
            hidden_features,
            hidden_layers=1,
            first_omega=30,
            hidden_omega=30,
            custom_init_function_=None,
    ):
        super().__init__()
        in_features = 2  # Input features (e.g., 2D coordinates for an image grid), input shape
        out_features = 1  # Output features (e.g., pixel intensities), output shape

        net = []

        # Add the first layer with custom omega and initialization for the first layer
        net.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                custom_init_function_=custom_init_function_,
                omega=first_omega,
            )
        )

        # Add the specified number of hidden layers with uniform omega
        for _ in range(hidden_layers):
            net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    custom_init_function_=custom_init_function_,
                    omega=hidden_omega,
                )
            )

        # Add the final linear layer to map hidden features to output features
        final_linear = nn.Linear(hidden_features, out_features)
        if custom_init_function_ is None:
            # Use the default SIREN initialization for the final layer
            paper_init_(final_linear.weight, is_first=False, omega=hidden_omega)
        else:
            # Use custom initialization if provided
            custom_init_function_(final_linear.weight)

        net.append(final_linear)

        # Store the network as a sequential model
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """
        Perform the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, 2)`, representing the 2D pixel coordinates.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(n_samples, 1)`, representing the predicted
            intensities or pixel values.
        """
        return self.net(x)


def generate_coordinates(n):
    """
    Generate a grid of 2D coordinates over a square domain [0, n) x [0, n).

    Parameters
    ----------
    n : int
        The number of points per dimension (e.g., n x n grid).

    Returns
    -------
    coords_abs : np.ndarray
        A 2D array of shape `(n**2, 2)`, where each row represents a pair of
        absolute (row, column) coordinates.

    Notes
    -----
    This function is typically used to generate input coordinates for SIREN
    networks, where each pixel coordinate in an image is treated as input.
    """
    # Generate a meshgrid of row and column indices
    # `indexing="ij"` ensures correct indexing order (rows, cols)
    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")

    # Flatten the grids and stack row and column indices as coordinate pairs
    coords_abs = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return coords_abs



def generate_coordinates(n):
    """Generate regular grid of 2D coordinates on [0, n] x [0, n].

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    coords_abs : np.ndarray
        Array of row and column coordinates of shape `(n ** 2, 2)`.
    """
    # Create a grid of row and column indices using np.meshgrid
    # `indexing="ij"` ensures that the first dimension corresponds to rows,
    # and the second dimension corresponds to columns
    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")

    # Flatten the row and column grids and stack them as pairs of coordinates
    coords_abs = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return coords_abs

class PixelDataset(torch.utils.data.Dataset):

    def __init__(self, clean_img, noisy_img):

        self.clean_img = clean_img
        self.noisy_img = noisy_img
        self.size = clean_img.shape[0]
        self.coords_abs = generate_coordinates(self.size)
        self.coords = torch.cartesian_prod(
            torch.arange(clean_img.shape[0]), torch.arange(clean_img.shape[1])
        )
        self.coords = self.coords.float() / clean_img.shape[0]  # Normalize to [0, 1]

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        coord = self.coords[idx]
        coords_abs = self.coords_abs[idx]
        r, c = coords_abs

        x, y = int(coord[0] * self.clean_img.shape[0]), int(coord[1] * self.clean_img.shape[1])

        return {
            "coords": coord,
            'coords_abs': coords_abs,
            "noisy_intensity": self.noisy_img[x, y],
            "clean_intensity": self.clean_img[x, y]
        }


class GradientUtils:
    @staticmethod
    def gradient(target, coords):
        """
        Compute the gradient of a target function with respect to input coordinates.

        Parameters
        ----------
        target : torch.Tensor
            A 2D tensor of shape `(n_coords, ?)` representing the target values.
            These could be scalar values or a vector field evaluated at the given coordinates.

        coords : torch.Tensor
            A 2D tensor of shape `(n_coords, 2)` representing the 2D spatial coordinates.

        Returns
        -------
        grad : torch.Tensor
            A 2D tensor of shape `(n_coords, 2)` representing the gradient of the 
            target with respect to the x and y coordinates.

        Notes
        -----
        - The function uses `torch.autograd.grad` to compute the derivatives, and 
          it enables the computation graph (`create_graph=True`) for higher-order derivatives.
        """
        return torch.autograd.grad(
            target,                     # The target function for which the gradient is computed
            coords,                     # The variables with respect to which the gradient is computed
            grad_outputs=torch.ones_like(target),  # Gradient initialization (same shape as the target)
            create_graph=True           # Allows further computation for higher-order derivatives
        )[0]  # Extract the result tensor from the returned tuple

    @staticmethod
    def divergence(grad, coords):
        """
        Compute the divergence of a vector field (grad) with respect to the coordinates.

        Parameters
        ----------
        grad : torch.Tensor
            A 2D tensor of shape `(n_coords, 2)` representing the gradient (vector field) 
            with respect to x and y.

        coords : torch.Tensor
            A 2D tensor of shape `(n_coords, 2)` representing the 2D spatial coordinates.

        Returns
        -------
        div : torch.Tensor
            A 2D tensor of shape `(n_coords, 1)` representing the divergence 
            of the input gradient field.

        Notes
        -----
        - In a 2D context, the divergence is the sum of second derivatives 
          (`∂²/∂x²` and `∂²/∂y²`).
        - This function loops over each dimension of the coordinates to compute 
          partial derivatives along x and y.
        """
        div = 0.0
        # Loop through each dimension of the coordinates (e.g., x and y for 2D)
        for i in range(coords.shape[1]):
            div += torch.autograd.grad(
                grad[..., i],           # Select the gradient component along dimension `i`
                coords,                 # The variables with respect to which the divergence is computed
                grad_outputs=torch.ones_like(grad[..., i]),  # Gradient initialization
                create_graph=True       # Enable further computations
            )[0][..., i : i + 1]        # Extract the diagonal (partial derivative along `i`)
        return div

    @staticmethod
    def laplace(target, coords):
        """
        Compute the Laplace operator (Laplacian) of a target function.

        Parameters
        ----------
        target : torch.Tensor
            A 2D tensor of shape `(n_coords, 1)` representing the target values.

        coords : torch.Tensor
            A 2D tensor of shape `(n_coords, 2)` representing the 2D spatial coordinates.

        Returns
        -------
        laplace : torch.Tensor
            A 2D tensor of shape `(n_coords, 1)` representing the Laplacian of the target.

        Notes
        -----
        - The Laplacian is the divergence of the gradient of the target function:
          `Δf = ∇²f = div(∇f) = ∂²f/∂x² + ∂²f/∂y²`.
        - This method uses the `gradient` method to compute the gradient first, 
          and then the `divergence` method to compute its divergence.
        """
        # Compute the gradient of the target function with respect to the coordinates
        grad = GradientUtils.gradient(target, coords)
        
        # Compute the divergence of the gradient (which gives the Laplacian)
        return GradientUtils.divergence(grad, coords)
