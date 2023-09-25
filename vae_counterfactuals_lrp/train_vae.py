import os
from tqdm import tqdm

# monai imports
from monai.networks.nets import VarAutoEncoder
from monai.utils import set_determinism

# pytorch imports
import torch
from torch.utils.data import DataLoader

# Local imports
from data_processing import get_data_splits, get_labels_and_clinical
from dataset import DDSMdataset, get_transforms
from utils import EarlyStopping, save_side_by_side, loss_function_vae

# Import Config from config.py
from config import config


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

print("\n###################################")
print("Train labels:", len(label_train))
print("Validation labels:", len(label_val))
print("Test labels:", len(label_test))

print("\n###################################")
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
autoencoder = VarAutoEncoder(spatial_dims=2,
        in_shape=(config.input_channels, config.patch_size, config.patch_size),
        out_channels=config.input_channels,
        latent_size=config.latent_dimension,
        channels=config.channels,
        strides=config.strides)

autoencoder.to(device)


# defining optimizers
optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=config.vae_learning_rate)

# defining early stopping
early_stopping_gen = EarlyStopping(patience=config.patience_early_stopping, verbose=True, path=os.path.join(config.model_save_dir, f"{config.fold_no}_checkpoint.pt"))

# train loop
epoch_train_loss_list = []
epoch_val_loss_list = []

flag = 1
for epoch in range(config.n_epochs):
    train_loss = 0
    val_loss = 0

    autoencoder.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch["image"].to(device)
        images = images.type("torch.cuda.FloatTensor")

        # clear gradients
        optimizer.zero_grad()

        # infer the current batch
        reconstructed_img, mu, logvar, latent = autoencoder(images)

        # calculating the loss
        loss = loss_function_vae(reconstructed_img, images, mu, logvar)
        train_loss += loss.item()

        # backward loss and next step
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            {
                "gen_loss": train_loss / (step + 1)
            }
        )
    epoch_train_loss_list.append(train_loss / len(train_loader))

    autoencoder.eval()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        images = images.type("torch.cuda.FloatTensor")

        # infer the current batch
        reconstructed_img, mu, logvar, latent = autoencoder(images)

        # calculating the loss
        loss = loss_function_vae(reconstructed_img, images, mu, logvar)
        val_loss += loss.item()

        progress_bar.set_postfix(
            {
                "gen_loss": val_loss / (step + 1)
            }
        )
    epoch_val_loss_list.append(val_loss / len(val_loader))

    early_stopping_gen((val_loss / len(val_loader)), autoencoder)

    if early_stopping_gen.early_stop:
        print("Early stopping")
        break

    # saving snippets while training
    save_side_by_side(images.detach().cpu().numpy()[0, ...],
                      reconstructed_img.detach().cpu().numpy()[0, ...],
                      os.path.join(config.check_data_snippets, f"image_{config.fold_no}_epoch_{epoch}.png"))

