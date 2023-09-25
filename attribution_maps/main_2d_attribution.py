import sys
import os
import configparser
import skimage
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import captum
from heatmaps import HeatmapGenerator

def preprocess_image(image_path):
    try:
        img = skimage.io.imread(image_path)
    except FileNotFoundError:
        print(f"{image_path} not found.")
        return None

    img = xrv.datasets.normalize(img, 255)
    img = img[:, :, 0]
    img = img[None, :, :]    
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)
    return img


if __name__ == "__main__":
  # Config Parser initialization
  config = configparser.ConfigParser()
  config.read('config_heatmap.ini')

  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda"

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

  model = torch.load(model_path).to(device)

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

