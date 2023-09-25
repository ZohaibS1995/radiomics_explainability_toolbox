import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import nrrd
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import LRP

# Import configurations
from config_counterfactual import config_counterfactuals
from config import config

# Monai model
from monai.networks.nets import VarAutoEncoder

# Local imports
from data_processing import get_data_splits, get_labels_and_clinical
from utils import save_side_by_side
from train_mlp_image_clinical import MLP

# Define GPU requirements
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to preprocess an image
def preprocess_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = cv2.resize(image, (config_counterfactuals.patch_size, config_counterfactuals.patch_size))
    image = np.transpose(image, (2, 1, 0))
    image = np.expand_dims(image, axis=0)
    return torch.Tensor(image).to(device).type('torch.cuda.FloatTensor')


def compute_lrp_explanations(model, input_image, target_class=1):
    """
    Compute LRP explanations for an input image.

    Args:
        model (nn.Module): The neural network model.
        input_image (torch.Tensor): The input image for which explanations are to be computed.
        target_class (int): The target class index for which to compute explanations (default is 1).

    Returns:
        lrp_explanations (torch.Tensor): LRP explanations for the input image.
    """
    # Create an instance of the LRP attribution method
    lrp = LRP(model)

    # Move the input image to the same device as the model
    input_image = input_image.to(model.device)

    # Perform LRP attribution for the target class
    lrp_explanations = lrp.attribute(input_image, target=target_class)

    return lrp_explanations


def plot_lrp_explanations(local_exp, local_exp_names, save_path=None):
    """
    Plot lrp explanations as a bar plot and save it as an image.

    Args:
        local_exp (list): List of explanation values.
        local_exp_names (list): List of corresponding explanation names.
        save_path (str): Path to save the plot as an image (optional).

    Returns:
        None
    """
    # Normalize local_exp values to make them visually comparable
    normalized_local_exp = np.abs(local_exp) / np.max(np.abs(local_exp))

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(local_exp_names, normalized_local_exp, color='skyblue')
    plt.xlabel('Normalized Explanation Value')
    plt.title('Local Explanations')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top

    # Save the plot as an image
    if save_path:
        plt.savefig(os.path.join(config_counterfactuals.counterfactuals_save_directory, save_path), bbox_inches='tight')

    # Display the plot
    #plt.show()

# Function to validate and generate counterfactuals
def validate_and_generate_counterfactuals_and_local_lrp(vae_model, mlp_model, img_path, img_label, clinical_var, clinical_var_names, target_class=1):
    local_exp = []
    local_exp_names = []

    # Reading the image
    image = nrrd.read(img_path)[0]
    x = preprocess_image(image)
    y = torch.tensor(img_label).to(device)
    z = torch.tensor(clinical_var).to(device)

    # Getting the latent space
    reconstruction, mu, logvar, latent = vae_model(x)

    # combining latent variables and clinical variables
    latent = torch.cat((latent.squeeze(), z)).unsqueeze(0).to(device)
    latent = latent.type(torch.cuda.FloatTensor)

    # Getting the prediction from MLP
    lrp = LRP(mlp_model)
    lrp_explanations = lrp.attribute(latent, target=target_class)

    # generating explanations
    swe_lrp = np.sum(np.abs((lrp_explanations.detach().cpu().numpy())[0, :config_counterfactuals.latent_dimension-1]))
    clinical_lrp = lrp_explanations.detach().cpu().numpy()[0, config_counterfactuals.latent_dimension:]

    # local explanations
    local_exp.append(swe_lrp)
    local_exp.extend(list(clinical_lrp))

    # local explanations name
    local_exp_names.append("SWE")
    local_exp_names.extend(clinical_var_names)

    # model prediction
    pred_mlp = mlp_model(latent)

    # getting the model predictions
    pred_sf = F.softmax(pred_mlp)[:, 1]
    pred_prob = pred_sf.detach().cpu().numpy()[0]
    pred_threshold = pred_sf.detach().cpu().numpy()[0] > config_counterfactuals.threshold

    # saving the local explanations
    pred_prob = str(pred_prob).replace(".", "_")
    plot_lrp_explanations(local_exp, local_exp_names, save_path=f"label_{y}_pred_prob_{pred_prob}_local_explanation.png")

    # Logging outputs for analysis
    print(f"True Label: {img_label}, Predicted Label: {pred_threshold}, Softmax Prediction: {pred_prob}")

    dzdxp = torch.autograd.grad(pred_sf, latent)[0]

    for thresh_lam in range(config_counterfactuals.lambda_start, config_counterfactuals.lambda_end, config_counterfactuals.lambda_step):
        latent_mod = latent + dzdxp * thresh_lam
        pred_mod = mlp_model(latent_mod)

        pred_sf_mod = F.softmax(pred_mod)[:, 1].detach().cpu().numpy()[0]
        counterfactual = vae_model.decode_forward(latent_mod[..., :config_counterfactuals.latent_dimension])

        # Visualize the counterfactual and the original image
        pred_counterfactual_str = str(pred_sf_mod).replace(".", "_")
        save_side_by_side(image,
                          np.transpose(counterfactual.detach().cpu().numpy()[0, ...], (1, 2, 0)),
                          os.path.join(config_counterfactuals.counterfactuals_save_directory, f"label_{y}_pred_prob_{pred_counterfactual_str}_counterfactual.png"))

    return

def generate_global_explanation(vae_model, mlp_model, val_imgs, clinical_val, clinical_var_names, target_class=1):
    # global explanations array
    global_exp = []
    global_exp_names = []

    # variable names
    global_exp_names.append("SWE")
    global_exp_names.extend(clinical_var_names)

    for idx in range(len(val_imgs)):
        print(f"{idx}/{len(val_imgs)} \n")

        # selecting the instance
        img_path = val_imgs[idx]
        clinical_var = clinical_val[idx]

        # local explanations array
        local_exp = []

        # Reading the image
        image = nrrd.read(img_path)[0]
        x = preprocess_image(image)
        z = torch.tensor(clinical_var).to(device)

        # Getting the latent space
        reconstruction, mu, logvar, latent = vae_model(x)

        # combining latent variables and clinical variables
        latent = torch.cat((latent.squeeze(), z)).unsqueeze(0).to(device)
        latent = latent.type(torch.cuda.FloatTensor)

        # Getting the prediction from MLP
        lrp = LRP(mlp_model)
        lrp_explanations = lrp.attribute(latent, target=target_class)

        # generating explanations
        swe_lrp = np.sum(np.abs((lrp_explanations.detach().cpu().numpy())[0, :config_counterfactuals.latent_dimension-1]))
        clinical_lrp = lrp_explanations.detach().cpu().numpy()[0, config_counterfactuals.latent_dimension:]

        # local explanations
        local_exp.append(swe_lrp)
        local_exp.extend(list(clinical_lrp))

        # appending global explanations
        global_exp.append(local_exp)

    global_exp = np.sum(np.vstack(global_exp), axis=0)

    # saving the global explanations
    plot_lrp_explanations(global_exp, global_exp_names, save_path=f"lrp_global_explanation.png")

    return


# Define the VAE model
autoencoder = VarAutoEncoder(spatial_dims=2,
        in_shape=(config_counterfactuals.input_channels, config_counterfactuals.patch_size, config_counterfactuals.patch_size),
        out_channels=config_counterfactuals.input_channels,
        latent_size=config_counterfactuals.latent_dimension,
        channels=config_counterfactuals.channels,
        strides=config_counterfactuals.strides)
vae_model = autoencoder.to(device)

# Load the checkpoint for the VAE model
checkpoint = torch.load(config_counterfactuals.vae_model_path)
vae_model.load_state_dict(checkpoint)
vae_model.eval()

# Define the MLP model
mlp_model = MLP()
mlp_model.load_state_dict(torch.load(config_counterfactuals.mlp_model_path))
mlp_model.to(device)
mlp_model.eval()

# Using data paths from the Config
train_data_path, test_data_path, base_dir = config_counterfactuals.train_data_path, config_counterfactuals.test_data_path, config_counterfactuals.base_dir

# Extracting labels and clinical data
train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical = get_labels_and_clinical(base_dir)

# Splitting data into train, val, test
train_imgs, val_imgs, test_imgs, label_train, label_val, label_test, clinical_train, clinical_val, clinical_test = get_data_splits(train_data_path, test_data_path, train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical, config)

# reading the excel file
clinical_var_names = pd.read_csv(config_counterfactuals.path_clinical_data).columns.values[3:]

# Validate and generate counterfactuals
no_in_list = 0
if no_in_list > len(val_imgs):
    raise ValueError(f"The number {no_in_list} is out of range in the validation images list")
else:
    img_path = val_imgs[no_in_list]
    img_label = label_val[no_in_list]
    clinical_var = clinical_val[no_in_list]

    validate_and_generate_counterfactuals_and_local_lrp(vae_model, mlp_model, img_path, img_label, clinical_var, clinical_var_names)

# pass a list of images and corresponding labels and clinical variables
generate_global_explanation(vae_model, mlp_model, val_imgs, clinical_val, clinical_var_names, target_class=1)

