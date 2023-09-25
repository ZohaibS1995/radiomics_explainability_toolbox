import sys
import os
import configparser

import monai.networks.nets
import skimage
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import captum
from heatmaps import HeatmapGenerator
import SimpleITK as sitk
from monai.transforms import Compose, Resize, ToTensor

def preprocess_image(arr_path):
    arr_img = np.load(arr_path)
    val_transforms = Resize(spatial_size=(100, 256, 256))

    image_resized = val_transforms(np.expand_dims(arr_img, axis=0))
    img_tensor = torch.tensor(image_resized)
    img_tensor = img_tensor.type("torch.cuda.FloatTensor")
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def load_model(model_path):
    model = monai.networks.nets.resnet10().to(device)
    model.conv1 = torch.nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    checkpoint = torch.load(model_path)
    if 'module' in list(checkpoint.keys())[0]:
        new_state_dict = {k[7:]: v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    return model

if __name__ == "__main__":

  # setting up the GPU configuration
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = "4"

  # check if gpu is available
  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda"

  # Config Parser initialization
  config = configparser.ConfigParser()
  config.read('config_heatmap.ini')

  # heatmap generator config values
  method_type= config['HeatmapGenerator']['method_type']
  save_path_folder = config['HeatmapGenerator']['save_path_folder']
  target_layer = config['HeatmapGenerator']['target_layer']
  model_path = config['HeatmapGenerator']['model_path']

  # attribution config values
  target_val = int(config['Attributions']['target_val'])
  method = config['Attributions']['method']
  percentile = int(config['Attributions']['percentile'])
  alpha = float(config['Attributions']['alpha'])

  # image files
  image_names = config['ImageFiles']['image_names'].split(',')

  # loading the model checkpoint
  model = load_model(model_path)

  if config['Attributions']['method'] == "gradcam" or config['Attributions']['method'] == "guided_gradcam":
    exec("target_layer_model = model." + target_layer)
  else:
    target_layer_model = None

  heatmap_gen = HeatmapGenerator(model, 
                                  method_type=method_type, 
                                  target_layer=target_layer_model, 
                                  save_path_folder=save_path_folder)

  for image_name in image_names:
      img = preprocess_image(image_name)
      if img is None:
          continue
      
      try:
          res = heatmap_gen.generate_heatmap(img,
                                              method=config['Attributions']['method'],
                                              target_val=target_val)

          heatmap_gen.visualize_attributions(res,
                                              original_image=img, 
                                              method=method, 
                                              percentile=percentile,
                                              alpha=alpha,
                                              image_name = image_name)
      
      except Exception as e:
          print(f"Failed to generate heatmap: {str(e)}")


