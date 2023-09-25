# imports
import os
import numpy as np
from tqdm import tqdm

# monai imports
from monai.utils import set_determinism
from monai.networks.nets import VarAutoEncoder

# pytorch imports
import torch
from torch.utils.data import DataLoader

# Local imports
from data_processing import get_data_splits, get_labels_and_clinical
from dataset import DDSMdataset, get_transforms

# Import Config from config.py
from config import config

set_determinism(42)


# Using data paths from the Config
train_data_path, test_data_path, base_dir = config.train_data_path, config.test_data_path, config.base_dir

# Extracting labels and clinical data
train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical = get_labels_and_clinical(base_dir)


# setting up the device
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

dict_fold_latent_train = {}
dict_fold_latent_val = {}

dict_fold_label_train = {}
dict_fold_label_val = {}


# setting the fold number
fold_no = config.fold_no

# Splitting data into train, val, test
train_imgs, val_imgs, test_imgs, label_train, label_val, label_test, clinical_train, clinical_val, clinical_test = get_data_splits(
    train_data_path, test_data_path, train_labels_patientid, train_labels, test_labels_patientid, test_labels,
    train_clinical, test_clinical, config)

val_transform = get_transforms("val")
validation_dataset = DDSMdataset(image_paths=train_imgs, labels=label_train, clinical=clinical_train, transform=val_transform)
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)

# You can further include your training loop, model initialization, and other processes here.
print("Training Images:", len(train_imgs))
print("Validation Images:", len(val_imgs))
print("Test Images:", len(test_imgs))


# Define Autoencoder KL network
autoencoder = VarAutoEncoder(spatial_dims=2,
                             in_shape=(config.input_channels, config.patch_size, config.patch_size),
                             out_channels=config.input_channels,
                             latent_size=config.latent_dimension,
                             channels=config.channels,
                             strides=config.strides)
autoencoder.to(device)

# loading the model
model = torch.load(os.path.join(config.model_save_dir, f"{fold_no}_checkpoint.pt"))
autoencoder.load_state_dict(model)

# testing the model
progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
progress_bar.set_description(f"Generating Latent Representation for training images")

# generating latent representation
autoencoder.eval()

# adding lists
dict_fold_latent_train[fold_no] = []
dict_fold_label_train[fold_no] = []
count = 0
for step, batch in progress_bar:
    images = batch["image"].to(device).type("torch.cuda.FloatTensor")
    labels = batch["label"].detach().squeeze().numpy()
    clinical = batch["clinical"].detach().squeeze().numpy()

    # Generator part
    reconstruction, mu, logvar, latent = autoencoder(images)
    latent = latent.detach().cpu().squeeze().numpy().flatten()
    latent = np.concatenate((latent, clinical))

    # adding to the list
    dict_fold_latent_train[fold_no].append(latent)
    dict_fold_label_train[fold_no].append(labels[np.newaxis][0])
    count += 1

    if count > 40:
        break

np.save("latent_image_dict_train", dict_fold_latent_train)
np.save("latent_label_dict_train", dict_fold_label_train)

val_transform = get_transforms("val")
validation_dataset = DDSMdataset(image_paths=val_imgs, labels=label_val, clinical=clinical_val, transform=val_transform)
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)

# You can further include your training loop, model initialization, and other processes here.
print("Training Images:", len(train_imgs))
print("Validation Images:", len(val_imgs))
print("Test Images:", len(test_imgs))


# Define Autoencoder KL network
autoencoder = VarAutoEncoder(spatial_dims=2,
                             in_shape=(config.input_channels, config.patch_size, config.patch_size),
                             out_channels=config.input_channels,
                             latent_size=config.latent_dimension,
                             channels=config.channels,
                             strides=config.strides)
autoencoder.to(device)

# loading the model
model = torch.load(os.path.join(config.model_save_dir, f"{fold_no}_checkpoint.pt"))
autoencoder.load_state_dict(model)

# testing the model
progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
progress_bar.set_description(f"Generating Latent Representation for testing images")

# generating latent representation
autoencoder.eval()

# adding lists
dict_fold_latent_val[fold_no] = []
dict_fold_label_val[fold_no] = []

for step, batch in progress_bar:
    images = batch["image"].to(device).type("torch.cuda.FloatTensor")
    labels = batch["label"].detach().squeeze().numpy()
    clinical = batch["clinical"].detach().squeeze().numpy()

    # Generator part
    reconstruction, mu, logvar, latent = autoencoder(images)
    latent = latent.detach().cpu().squeeze().numpy().flatten()
    latent = np.concatenate((latent, clinical))

    # adding to the list
    dict_fold_latent_val[fold_no].append(latent)
    dict_fold_label_val[fold_no].append(labels[np.newaxis][0])


np.save("latent_image_dict_val", dict_fold_latent_val)
np.save("latent_label_dict_val", dict_fold_label_val)
