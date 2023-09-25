import numpy as np
import cv2
import nrrd
from torch.utils.data import Dataset
from monai.transforms import Affine, Compose, Spacing, ToTensor
from config import config
class DDSMdataset(Dataset):
    """
    Dataset class for loading and processing DDSM data.
    """
    def __init__(self, image_paths: list, labels: np.ndarray, clinical: np.ndarray, transform=None):
        """
        Initialize the DDSM dataset.

        Args:
        - image_paths (list): Paths to the images in the dataset.
        - labels (np.ndarray): Corresponding labels for the images.
        - clinical (np.ndarray): clinical values for the images.
        - transform (callable, optional): Optional transform to be applied on the images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.clinical = clinical
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset.

        Args:
        - idx (int): Index of the item.

        Returns:
        - dict: A dictionary containing the image, label, and clinical value.
        """
        image_filepath = self.image_paths[idx]
        label = self.labels[idx]
        clinical_value = self.clinical[idx]

        # Read the image
        image = nrrd.read(image_filepath)[0]
        image = self._normalize_and_resize(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        image = np.transpose(image, (2, 0, 1))

        return {"image": image, "label": label, "clinical": clinical_value}

    def _normalize_and_resize(self, image: np.ndarray):
        """
        Normalize and resize an image.

        Args:
        - image (np.ndarray): The image to be processed.

        Returns:
        - np.ndarray: The processed image.
        """
        # Normalize
        min_pixel = np.min(image)
        max_pixel = np.max(image)
        image = (image - min_pixel) / (max_pixel - min_pixel)

        # Resize
        image = cv2.resize(image, (config.patch_size, config.patch_size))
        return image

def get_transforms(phase: str):
    """
    Get transformations based on the phase.

    Args:
    - phase (str): Current phase (train, val, test).

    Returns:
    - Callable: Transformation function.
    """
    if phase == "train":
        return Compose([]) #A.Compose([
            #A.Affine(scale=(0.9, 1.1), always_apply=False, p=0.5),
            #A.Affine(translate_percent=0.05, always_apply=False, p=0.5),
            #A.Affine(rotate=(-20, 20), always_apply=False, p=0.5),
            #A.HorizontalFlip(p=0.5)
        #])
    else:
        return Compose([])

