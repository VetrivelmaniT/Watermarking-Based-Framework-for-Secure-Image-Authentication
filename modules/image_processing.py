import os
import cv2
import torch
import numpy as np
import logging

from .utils import load_image, save_image, preprocess_watermark
from watermark_localization import embed_copyright_watermark, extract_copyright_watermark
from watermark_copyright import embed_localization_watermark, extract_localization_watermark

from ppe_module import PPEM  
from bit_encryption import BitEncryptionModule, encrypt_watermark
from recovery_module import RecoveryModule, recover_watermark
from .config import OUTPUT_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_image(image_path):
    """Processes an image by embedding watermarks, enhancing, encrypting, and recovering."""
    try:
        logging.info(f"Processing image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

        # Step 1: Embed Localization Watermark

        
        # Ask user for watermark text
        watermark_text = input("Enter watermark text (e.g., TamperCheck123): ")

        # Define path where watermarked image will be saved
        copyright_path = os.path.join(OUTPUT_FOLDER, "copyright_watermarked.png")

        # Call function to embed watermark
        embed_copyright_watermark(image_path, watermark_text, copyright_path)

        # Log the result
        logging.info("Copyright watermark embedded.")

        # Step 2: Embed Copyright Watermark
        loc_watermarked_path = os.path.join(OUTPUT_FOLDER, "localization_watermarked.png")
        embed_localization_watermark(image_path, loc_watermarked_path)
        logging.info("Localization watermark embedded.")

        # Step 3:  Extract Copyright Watermark
        extracted_text = extract_copyright_watermark(copyright_path)
        logging.info(f"Extracted copyright watermark: {extracted_text}")

        # Step 4: Extract Localization Watermark
        extracted_path = os.path.join(OUTPUT_FOLDER, "extracted_localization_watermark.png")
        extract_localization_watermark(image_path, loc_watermarked_path, extracted_path)
        logging.info("Extracted localization_watermark.")

        # Step 5: Enhance using PPEM model
        ppe_model = PPEM()

        # Ensure model is in eval mode
        ppe_model.eval()

        # Load the image from extracted path
        image_np = load_image(extracted_path)

        # Validate if image was loaded
        if image_np is None:
            raise ValueError(f"Failed to load image from {extracted_path}")
        else:
            logging.info(f"Image loaded successfully: shape={image_np.shape}, dtype={image_np.dtype}")

        # Validate image dimensions
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), but got shape: {image_np.shape}")

        # Convert image from BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Convert to torch tensor (C, H, W) format
        image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Check input tensor values
        logging.info(f"Input tensor stats - min: {image_tensor.min()}, max: {image_tensor.max()}")

        # Run through PPEM model without gradients
        with torch.no_grad():
            enhanced_tensor = ppe_model(image_tensor)

        # Check output tensor shape
        if enhanced_tensor.shape != image_tensor.shape:
            logging.warning(f"Unexpected output shape from PPEM: {enhanced_tensor.shape}")

        # Convert torch tensor back to numpy
        enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced_np = (enhanced_np * 255).clip(0, 255).astype(np.uint8)

        # Display enhanced image shape and pixel range for validation
        logging.info(f"Enhanced image stats - shape: {enhanced_np.shape}, dtype: {enhanced_np.dtype}, min: {enhanced_np.min()}, max: {enhanced_np.max()}")

        # Convert RGB to BGR for OpenCV saving
        enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

        # Save the enhanced image
        enhanced_output_path = os.path.join(OUTPUT_FOLDER, "enhanced_output.png")
        cv2.imwrite(enhanced_output_path, enhanced_bgr)

        # Confirm successful save
        if os.path.exists(enhanced_output_path):
            logging.info(f"Enhanced image saved successfully at: {enhanced_output_path}")
        else:
            logging.error("Failed to save enhanced image.")

        # Step 6: Encrypt and Recover Watermark
        encrypt_and_recover_watermark()

    except Exception as e:
        logging.error(f"Error processing image '{image_path}': {e}")

def encrypt_and_recover_watermark():
    """Encrypts and recovers a watermark using neural network models."""
    try:
        input_size, hidden_size, output_size = 10, 20, 10
        encryption_model = BitEncryptionModule(input_size, hidden_size, output_size)
        recovery_model = RecoveryModule(output_size, hidden_size, input_size)

        watermark_str = "1010101010"
        watermark_tensor = preprocess_watermark(watermark_str)

        encrypted = encrypt_watermark(watermark_tensor, encryption_model)
        recovered = recover_watermark(encrypted, recovery_model)

        logging.info(f"Original Watermark: {watermark_str}")
        logging.info(f"Recovered Watermark: {recovered.detach().numpy()}")

    except Exception as e:
        logging.error(f"Error in encryption & recovery: {e}")
