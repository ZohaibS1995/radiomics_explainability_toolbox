# Post-Hepatectomy Liver Failure Prediction with Counterfactual Explanations and Layerwise Relevance Propagation (LRP)

## Overview

This tool is designed for post-hepatectomy liver failure prediction using Shear Wave Elastography (SWE) images. It incorporates a multi-step process, including the training of a Variational Autoencoder (VAE) to create a latent space representation of the images, training a Multi-Layer Perceptron (MLP) classifier using both the latent space and clinical variables, and generating counterfactual explanations. Additionally, it provides Layerwise Relevance Propagation (LRP) local and global plots to interpret the model's predictions.

## Getting Started

Before using this tool, make sure you have the necessary prerequisites in place:

### Installation

Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use this tool for liver failure prediction and generate counterfactual explanations and LRP plots, follow these steps:

### Data Preparation

Update the data paths and parameters in the configuration files (`config_counterfactuals.py` and `config.py`) to point to your dataset and set the necessary parameters for training and inference.

### Training

1. Train the Variational Autoencoder (VAE) to create a latent space representation of the SWE images. Run the following command:

   ```bash
   python train_vae.py
   ```

2. Train the Multi-Layer Perceptron (MLP) classifier using the latent space representations and clinical variables. Run the following command:

   ```bash
   python train_mlp_image_clinical.py
   ```

### Prediction and Counterfactual Generation

1.  Update the `test_data_path` parameter in `config_counterfactuals.py` to specify the path to the test data.


2. Generate counterfactual explanations for the predictions to highlight relevant areas of the image and suggest changes to prevent liver failure. Run the following command:

   ```bash
   python generate_counterfactual_lrp.py
   ```

### Layerwise Relevance Propagation (LRP) Explanations

1. The tool provides the capability to compute LRP explanations for input images using the LRP method. You can use the `compute_lrp_explanations` function to compute LRP explanations for a specific image.

2. Use the `plot_lrp_explanations` function to visualize and save LRP explanations as bar plots. These explanations highlight the contribution of SWE images and clinical variables to the model's predictions.

### Interpretation

1. Examine the generated counterfactual explanations to gain insights into the model's predictions. These explanations not only pinpoint the relevant areas of the image but also suggest alterations needed to avoid liver failure.

2. Utilize the LRP explanations to understand the contribution of SWE images and clinical variables to the model's predictions. These explanations provide a clear breakdown of feature relevance.

## Customization

You can customize the tool's behavior by modifying the configuration files (`config_counterfactuals.py` and `config.py`). Adjust parameters such as model hyperparameters, data paths, and visualization settings to suit your specific use case.

## Results

The tool provides predictions for post-hepatectomy liver failure, generates counterfactual explanations, and offers LRP explanations to enhance model interpretability. The counterfactual explanations highlight relevant image areas and suggest changes that can be made to reduce the risk of liver failure. LRP explanations provide a detailed breakdown of feature relevance.
