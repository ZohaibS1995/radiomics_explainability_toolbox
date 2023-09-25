
# Post-Hepatectomy Liver Failure Prediction with Counterfactual Explanations

## Overview

This tool is designed for predicting post-hepatectomy liver failure using Shear Wave Elastography (SWE) images. It leverages a combination of techniques, including a Variational Autoencoder (VAE) to create a latent space representation of the images and a Multi-Layer Perceptron (MLP) classifier to predict the occurrence of liver failure. Additionally, it generates counterfactual explanations that not only highlight the relevant areas of the image but also demonstrate the changes required to prevent liver failure.

## Getting Started

Before using this tool, make sure you have the necessary prerequisites in place:

### Installation

Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

### Training

1. Train the Variational Autoencoder (VAE) to create a latent space representation of the SWE images. Run the following command:

   ```bash
   python train_vae.py
   ```
   
2. Generate latent space of the train, validation and the test images. The training specific parameters can be configured using config.ini file.

   ```bash
   python generate_latent_vae.py
   ```

3. Train the Multi-Layer Perceptron (MLP) classifier using the latent space representations and corresponding labels. Run the following command:

   ```bash
   python train_mlp_image_only.py
   ```

### Prediction and Counterfactual Generation

Generate counterfactual explanations for the predictions to highlight relevant areas of the image and suggest changes to prevent liver failure. Run the following command:

   ```bash
   python generate_counterfactuals_image.py
   ```

### Interpretation

1. Examine the generated counterfactual explanations to gain insights into the model's predictions. These explanations not only pinpoint the relevant areas of the image but also suggest alterations needed to avoid liver failure.

2. Use these insights for clinical decision-making.

## Customization

You can customize the tool's behavior by modifying the configuration file (`config.ini`). Adjust parameters such as model hyperparameters, data paths, and visualization settings to suit your specific use case.

## Results

The tool provides predictions for post-hepatectomy liver failure and generates counterfactual explanations, offering valuable insights into the model's decision-making process. The counterfactual explanations highlight relevant image areas and suggest changes that can be made to reduce the risk of liver failure.


