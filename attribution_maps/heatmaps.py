# Base Python packages
import os
from pathlib import Path
from io import BytesIO

# External packages for handling arrays and images
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# PyTorch and its related packages for deep learning and image transformation
import torch  
from torch import device as torch_device  # For specifying the computation device
from torch.nn import Module  # Base class for all neural network modules
from torch.nn.functional import interpolate  # Function for resizing images
from torchvision.transforms import functional as F  # Provides several transform functions that can be chained together using `Compose`

# Captum package for model interpretability
from captum.attr import IntegratedGradients, GuidedBackprop, LayerGradCam, GuidedGradCam, InputXGradient  # Algorithms for feature attributions
from captum.attr import visualization as viz  # Visualization tools for interpreting neural networks


class HeatmapGenerator:
    """
    Class to generate attributions of model's prediction on given input and visualize them as heatmaps.
    """

    def __init__(self, model, method_type = "2D", target_layer=None, save_path_folder = "results"):
        """
        Initialize HeatmapGenerator with a trained model, method type, and target layer.

        :param model: a trained model, instance of torch.nn.Module.
        :param method_type: method type, either "2D" or "3D".
        :param target_layer: target layer for LayerGradCam and GuidedGradCam methods.
        :param save_path_folder: folder path to save results.
        """
        # Ensure the model is not None and of instance torch.nn.Module
        assert model is not None, "Model cannot be None."
        assert method_type in ['2D', '3D'], "Invalid method. Method should be '2D' or '3D'"


        self.model = model
        self.target_layer = target_layer
        self.integrated_gradients = IntegratedGradients(self.model)
        self.input_x_gradient = InputXGradient(self.model)
        self.guided_backprop = GuidedBackprop(self.model)
        self.guided_gradcam = GuidedGradCam(self.model, self.target_layer)
        self.gradcam = LayerGradCam(self.model, self.target_layer)
        self.method_type = method_type
        self.device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.zero_grad()
        self.default_cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
        self.save_path_folder = save_path_folder
        save_dir = Path(self.save_path_folder).parent
        os.makedirs(save_dir, exist_ok=True)


    def generate_heatmap(self, input_image, method='integrated_gradients', target_val = 1):
        """
        Generate heatmap for given input image.

        :param input_image: a PyTorch tensor, input image to compute attributions.
        :param method: method to compute attributions. It should be one of integrated_gradients, input_x_gradient, guided_backprop, guided_gradcam, or gradcam.
        :param attr_type: type of attributions, either "2D" or "3D".
        :param num_channels: number of channels in the image.
        :param target_val: target class index for which attributions are computed.
        :return: attributions for the given input image.
        """

        # Ensure the input image is a PyTorch tensor and it has 4 dimensions.
        methods = ['integrated_gradients', 'input_x_gradient', 'guided_backprop', 'guided_gradcam', 'gradcam']

        assert method in methods, f"Invalid method. Method should be one of {methods}"
        if method == 'gradcam':
          assert self.target_layer is not None, "target layer is not specified during class initialise. Please initialise the class again."

        input_image = input_image.to(self.device)
        input_image.requires_grad = True

        target = target_val
        if method == 'integrated_gradients':
            attributions = self.integrated_gradients.attribute(input_image, target=target, baselines=input_image * 0, internal_batch_size=2)
        elif method == 'input_x_gradient':
            attributions = self.input_x_gradient.attribute(input_image, target=target)
        elif method == 'guided_backprop':
            attributions = self.guided_backprop.attribute(input_image, target=target)
        elif method == 'guided_gradcam':
            attributions = self.guided_gradcam.attribute(input_image, target=target)
        elif method == 'gradcam':
            attributions = self.gradcam.attribute(input_image, target=target)

        if method == "guided_gradcam" or method == "gradcam":
            attributions = interpolate(attributions, size=input_image.shape, mode='bilinear', align_corners=False)

        return attributions


    def _cumulative_sum_threshold(self, values, percentile):
        """
        Compute cumulative sum threshold of given values at a certain percentile.

        :param values: values to compute threshold.
        :param percentile: percentile to compute threshold.
        :return: cumulative sum threshold.
        """
        # Ensure percentile is between 0 and 100
        assert percentile >= 0 and percentile <= 100, (
            "Percentile for thresholding must be " "between 0 and 100 inclusive."
        )

        sorted_vals = np.sort(values.flatten())
        cum_sums = np.cumsum(sorted_vals)
        threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
        return sorted_vals[threshold_id]


    def _normalize_scale(self, attr, scale_factor):
        """
        Normalizes the attribute values based on the scale factor. The resulting normalized attributes
        are clipped between -1 and 1.

        Args:
            attr (numpy.ndarray): Array of attributes to be normalized.
            scale_factor (float): The factor to scale the attributes by.

        Returns:
            numpy.ndarray: Normalized attribute array.
        """
        assert scale_factor != 0, "Cannot normalize by scale factor = 0"
        attr_norm = attr / scale_factor
        return np.clip(attr_norm, -1, 1)


    def save_image(self, plt, path):
        """
        Saves a generated matplotlib plot as an image file.

        Args:
            plt (matplotlib.pyplot): A matplotlib plot.
            path (str): The full path for the image file to be saved.
        """
        fig = plt.gcf()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_array = np.array(canvas.renderer.buffer_rgba())
        Image.fromarray(img_array).save(path)


    def visualize_attributions(self, attributions, original_image=None, method='method_name', percentile=98, alpha=0.5, image_name="inp1"):
            """
            Visualizes the attributions by generating heatmaps and overlaying them on the original image.

            Args:
                attributions (Tensor): The attributions generated by a model for an image.
                original_image (Tensor, optional): The original input image. Defaults to None.
                method (str, optional): The method used to generate attributions. Defaults to 'method_name'.
                percentile (int, optional): The percentile of the sum of absolute values of attributions to use as a threshold for visualization. Defaults to 98.
                alpha (float, optional): The alpha value for the colormap of the overlayed image. Defaults to 0.5.
                image_name (str, optional): The name of the image for which attributions are visualized. Defaults to "".
            """

            image_name = image_name.split(".")[0]
            # Ensure save directory exists
            os.makedirs(self.save_path_folder, exist_ok=True)
            attributions = attributions.squeeze().detach().cpu().numpy()
            original_image = original_image.squeeze().detach().cpu().numpy()

            if len(attributions.shape) == 3:
                if self.method_type == "2D":
                    attributions = np.transpose(attributions, (1,2,0))
                    attributions_tmp = np.sum(attributions, axis=2)

                    threshold = self._cumulative_sum_threshold(np.abs(attributions_tmp), percentile)
                    attributions_norm = self._normalize_scale(attributions_tmp, threshold)

                    plt.figure(figsize=(10,10))
                    plt.imshow(attributions_norm, cmap=self.default_cmap, vmin=-1, vmax=1)
                    self.save_image(plt, os.path.join(self.save_path_folder, image_name + "_" + method + "_heatmap.png"))

                    plt.figure(figsize=(10,10))
                    plt.imshow(original_image, alpha=1, cmap="gray")
                    plt.imshow(attributions_norm, cmap=self.default_cmap, vmin=-1, vmax=1, alpha=1-alpha)
                    self.save_image(plt, os.path.join(self.save_path_folder,  image_name + "_" + method + "_overlay.png"))
                    plt.show()

                else:  # for 3D
                    #print("grid_dims:", int(np.ceil(np.sqrt(original_image.shape[0]))))
                    #print(original_image.shape)
                    #print(attributions.shape)
                    # Grid dimensions
                    grid_dims = int(np.ceil(np.sqrt(original_image.shape[0])))

                    # threshold attributions
                    threshold = self._cumulative_sum_threshold(np.abs(attributions), percentile)
                    attributions_norm = self._normalize_scale(attributions, threshold)

                    fig, axs = plt.subplots(grid_dims, grid_dims, figsize=(20, 20))

                    # Plotting heatmaps
                    for i in range(grid_dims):
                        for j in range(grid_dims):
                            ax = axs[i, j]
                            ax.axis('off')
                            slice_index = i * grid_dims + j
                            if slice_index < original_image.shape[0]:
                                attr_slice = attributions_norm[slice_index, :, :]
                                ax.imshow(attr_slice, cmap=self.default_cmap, vmin=-1, vmax=1)

                    plt.tight_layout()
                    #plt.show()
                    fig.savefig(os.path.join(self.save_path_folder, image_name + "_" + method + "_heatmap_grid.png"), dpi=300)

                    # Plotting overlays
                    fig, axs = plt.subplots(grid_dims, grid_dims, figsize=(20, 20))

                    for i in range(grid_dims):
                        for j in range(grid_dims):
                            ax = axs[i, j]
                            ax.axis('off')
                            slice_index = i * grid_dims + j
                            if slice_index < original_image.shape[0]:
                                attr_slice = attributions_norm[slice_index, :, :]
                                original_slice = original_image[slice_index, :, :]

                                ax.imshow(original_slice, alpha=1, cmap="gray")
                                im = ax.imshow(attr_slice, cmap=self.default_cmap, vmin=-1, vmax=1, alpha=1-alpha)

                                # Colorbar
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="5%", pad=0.05)
                                plt.colorbar(im, cax=cax, orientation='vertical')

                    plt.tight_layout()
                    #plt.show()
                    fig.savefig(os.path.join(self.save_path_folder,  image_name + "_" + method + "_overlay_grid.png"), dpi=300)
            else:
                threshold = self._cumulative_sum_threshold(np.abs(attributions), percentile)
                attributions_norm = self._normalize_scale(attributions, threshold)

                plt.figure(figsize=(10,10))
                plt.imshow(attributions_norm, cmap=self.default_cmap, vmin=-1, vmax=1)
                self.save_image(plt, os.path.join(self.save_path_folder,  image_name + "_" + method + "_heatmap.png"))

                plt.figure(figsize=(10,10))
                plt.imshow(original_image, alpha=1, cmap="gray")
                plt.imshow(attributions_norm, cmap=self.default_cmap, vmin=-1, vmax=1, alpha=1-alpha)
                self.save_image(plt, os.path.join(self.save_path_folder, image_name + "_" + method + "_overlay.png"))
                plt.show()

    def compute_and_visualize_attributions(self, image_list, method='integrated_gradients', attr_type="2D", num_channels="3", target_val=38, percentile=98, alpha=0.5):
        """
        Compute attributions and visualize them as heatmaps for a list of input images.

        :param image_list: list of input images.
        :param method: method to compute attributions.
        :param attr_type: type of attributions, either "2D" or "3D".
        :param num_channels: number of channels in the image.
        :param target_val: target class index for which attributions are computed.
        :param percentile: percentile to compute threshold.
        :param alpha: alpha value for overlaying heatmap on the original image.
        """
        for i, image in enumerate(image_list):
            # Compute attributions
            attributions = self.generate_heatmap(image, method, attr_type, num_channels, target_val)

            # Visualize attributions
            image_name = f'image_{i}'
            self.visualize_attributions(attributions, image, method, percentile, alpha, image_name)