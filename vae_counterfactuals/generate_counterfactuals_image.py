import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import nrrd

# Import configurations
from config_counterfactual import config_counterfactuals
from train_mlp_image_only import MLP_only_image
from config import config

# Monai model
from monai.networks.nets import VarAutoEncoder

# Local imports
from data_processing import get_data_splits, get_labels_and_clinical
from utils import save_side_by_side

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

# Function to validate and generate counterfactuals
def validate_and_generate_counterfactuals(vae_model, mlp_model, img_path, img_label):
    # Reading the image
    image = nrrd.read(img_path)[0]
    x = preprocess_image(image)
    y = torch.tensor(img_label).to(device)

    # Getting the latent space
    reconstruction, mu, logvar, latent = vae_model(x)

    # Getting the prediction from MLP
    pred_mlp = mlp_model(latent)
    pred_sf = F.softmax(pred_mlp)[:, 1]
    pred_prob = pred_sf.detach().cpu().numpy()[0]
    pred_threshold = pred_sf.detach().cpu().numpy()[0] > config_counterfactuals.threshold

    # Logging outputs for analysis
    print(f"True Label: {img_label}, Predicted Label: {pred_threshold}, Softmax Prediction: {pred_prob}")

    dzdxp = torch.autograd.grad(pred_sf, latent)[0]

    for thresh_lam in range(config_counterfactuals.lambda_start, config_counterfactuals.lambda_end, config_counterfactuals.lambda_step):
        latent_mod = latent + dzdxp * thresh_lam
        pred_mod = mlp_model(latent_mod)

        pred_sf_mod = F.softmax(pred_mod)[:, 1].detach().cpu().numpy()[0]
        counterfactual = vae_model.decode_forward(latent_mod)

        # Visualize the counterfactual and the original image
        pred_counterfactual_str = str(pred_sf_mod).replace(".", "_")
        save_side_by_side(image,
                          np.transpose(counterfactual.detach().cpu().numpy()[0, ...], (1, 2, 0)),
                          os.path.join(config_counterfactuals.counterfactuals_save_directory, f"label_{y}_pred_prob_{pred_counterfactual_str}_counterfactual.png"))

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
mlp_model = MLP_only_image()
mlp_model.load_state_dict(torch.load(config_counterfactuals.mlp_model_path))
mlp_model.to(device)
mlp_model.eval()

# Using data paths from the Config
train_data_path, test_data_path, base_dir = config_counterfactuals.train_data_path, config_counterfactuals.test_data_path, config_counterfactuals.base_dir

# Extracting labels and clinical data
train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical = get_labels_and_clinical(base_dir)

# Splitting data into train, val, test
train_imgs, val_imgs, test_imgs, label_train, label_val, label_test, clinical_train, clinical_val, clinical_test = get_data_splits(train_data_path, test_data_path, train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical, config)

# Validate and generate counterfactuals
no_in_list = 0
if no_in_list > len(val_imgs):
    raise ValueError(f"The number {no_in_list} is out of range in the validation images list")
else:
    sel_img_path = val_imgs[no_in_list]
    sel_img_label = label_val[no_in_list]

    validate_and_generate_counterfactuals(vae_model, mlp_model, sel_img_path, sel_img_label)
