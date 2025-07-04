import os
import cv2
import numpy as np
import logging
import torch

def load_image(path):
    """Load an image from file path."""
    try:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Error loading image: {str(e)}")
        return None

def save_image(image, path):
    """Save an image to file path."""
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if cv2.imwrite(path, image):
            logging.info(f"Saved image to: {os.path.abspath(path)}")
            return os.path.abspath(path)
        else:
            raise IOError(f"Failed to write image to {path}")
    except Exception as e:
        logging.error(f"Error saving image: {str(e)}")
        return None

def preprocess_watermark(watermark_str):
    """Convert watermark string to tensor."""
    return torch.tensor([float(c) for c in watermark_str], dtype=torch.float32)