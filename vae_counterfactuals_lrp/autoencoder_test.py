import os
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm

# monai imports
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import set_determinism

# monai generative models
from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.nn import L1Loss

# Local imports
from data_processing import get_data_splits, get_labels_and_clinical
from dataset import DDSMdataset, get_transforms
from utils import EarlyStopping, KL_loss, save_side_by_side

# Import Config from config.py
from config import config
from sklearn.linear_model import LogisticRegression


# Using data paths from the Config
train_data_path, test_data_path, base_dir = config.train_data_path, config.test_data_path, config.base_dir

# Extracting labels and clinical data
train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical = get_labels_and_clinical(base_dir)

# Splitting data into train, val, test
train_imgs, val_imgs, test_imgs, label_train, label_val, label_test, clinical_train, clinical_val, clinical_test = get_data_splits(train_data_path, test_data_path, train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical, config)

# Creating datasets and dataloaders
train_transform = get_transforms("train")
val_transform = get_transforms("val")
test_transform = get_transforms("test")

training_dataset = DDSMdataset(image_paths=train_imgs, labels=label_train, clinical=clinical_train, transform=train_transform)
validation_dataset = DDSMdataset(image_paths=val_imgs, labels=label_val, clinical=clinical_val, transform=val_transform)
test_dataset = DDSMdataset(image_paths=test_imgs, labels=label_test, clinical=clinical_test, transform=test_transform)

train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)



print("Training Images:", len(train_imgs))
print("Validation Images:", len(val_imgs))
print("Test Images:", len(test_imgs))

print("\n")
print("Train labels:", len(label_train))
print("Validation labels:", len(label_val))
print("Test labels:", len(label_test))

print("\n")
print("Train clinical:", len(clinical_train))
print("Validation clinical:", len(clinical_val))
print("Test clinical:", len(clinical_test))

set_determinism(42)

# setting up the device
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# Define Autoencoder KL network
autoencoder = AutoencoderKL(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_channels=(32, 32, 64, 64),
    latent_channels=1,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True, True),
)
autoencoder.to(device)

# loading the model
model = torch.load(os.path.join(config.model_save_dir, f"{config.fold_no}_checkpoint.pt"))
autoencoder.load_state_dict(model)

# testing the model
progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), ncols=110)
progress_bar.set_description(f"Testing the model")

# generating latent representation
autoencoder.eval()
for step, batch in progress_bar:
    images = batch["image"].to(device).type("torch.cuda.FloatTensor")

    # Generator part
    reconstruction, z_mu, z_sigma = autoencoder(images)

    print(z_mu.shape)
    save_side_by_side(images.detach().cpu().numpy()[0,...],
                      reconstruction.detach().cpu().numpy()[0, ...],
                      os.path.join(config.check_data_test_snippets, f"image_{config.fold_no}_epoch_{step}.png"))

