# import os
# import cv2
# import numpy as np
# import logging
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from .utils import save_image
# from .config import OUTPUT_FOLDER
# from skimage.color import rgb2lab, deltaE_cie76
# from .utils import save_image
# from .config import OUTPUT_FOLDER

# def compare_images(original_path, tampered_path, output_folder="tamper_results"):
#     """Enhanced image comparison for detecting structural and color-based tampering"""
#     try:
#         # Load images
#         original = cv2.imread(original_path)
#         tampered = cv2.imread(tampered_path)

#         if original is None or tampered is None:
#             raise ValueError("Failed to load one or both images. Check the file paths.")

#         # Resize if dimensions don't match
#         if original.shape != tampered.shape:
#             tampered = cv2.resize(tampered, (original.shape[1], original.shape[0]))

#         # Convert to grayscale
#         original_gray = original
#         tampered_gray = tampered

#         # 1. Grayscale difference
#         diff_gray = cv2.absdiff(original_gray, tampered_gray)

#         # 2. RGB difference
#         diff_rgb = cv2.absdiff(original, tampered)

#         # Stack images horizontally
#         comparison = np.hstack((original_gray, tampered_gray))

#         # 3. Perceptual color difference (Delta E in LAB)
#         original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
#         tampered_lab = cv2.cvtColor(tampered, cv2.COLOR_BGR2LAB)
#         delta_e = np.linalg.norm(original_lab.astype("float") - tampered_lab.astype("float"), axis=2)
#         delta_e_norm = cv2.normalize(delta_e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         delta_e_color = cv2.applyColorMap(delta_e_norm, cv2.COLORMAP_JET)

#         # 4. SSIM and binary mask for highlighting
#         (ssim_score, diff_map) = ssim(original_gray, tampered_gray, full=True)
#         diff_map = (diff_map * 255).astype("uint8")
#         _, ssim_thresh = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#         # 5. Highlight changed regions using bounding boxes
#         highlight = tampered.copy()
#         contours, _ = cv2.findContours(ssim_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if cv2.contourArea(contour) > 40:
#                 (x, y, w, h) = cv2.boundingRect(contour)
#                 cv2.rectangle(highlight, (x, y), (x + w, y + h), (0, 0, 255), 2)

#         # 6. Final binary mask showing all detected changes (combined SSIM + grayscale)
#         final_binary = np.zeros_like(ssim_thresh)
#         final_binary[(ssim_thresh > 0) | (diff_gray > 10)] = 255

#         # 7. Crop region of change
#         changed_coords = cv2.findNonZero(final_binary)
#         if changed_coords is not None:
#             x, y, w, h = cv2.boundingRect(changed_coords)
#             cropped_changes = final_binary[y:y+h, x:x+w]
#         else:
#             cropped_changes = final_binary  # fallback: entire image
            
            

#         # Create output folder
#         os.makedirs(output_folder, exist_ok=True)

#         # Save outputs
#         cv2.imwrite(os.path.join(output_folder, "01_original.png"), original)
#         cv2.imwrite(os.path.join(output_folder, "02_tampered.png"), tampered)
#         cv2.imwrite(os.path.join(output_folder, "03_diff_gray.png"), diff_gray)
#         cv2.imwrite(os.path.join(output_folder, "04_diff_rgb.png"), diff_rgb)
#         cv2.imwrite(os.path.join(output_folder, "05_delta_e_gray.png"), delta_e_norm)
#         cv2.imwrite(os.path.join(output_folder, "06_delta_e_colormap.png"), delta_e_color)
#         cv2.imwrite(os.path.join(output_folder, "07_highlighted_diff.png"), highlight)
#         cv2.imwrite(os.path.join(output_folder, "08_ssim_thresh.png"), ssim_thresh)
#         cv2.imwrite(os.path.join(output_folder, "09_final_binary_output.png"), final_binary)
#         cv2.imwrite(os.path.join(output_folder, "10_cropped_changed_region.png"), cropped_changes)
#         cv2.imwrite(os.path.join(output_folder, "11_comparison_output.png"), comparison)
        
        
#                 # Step 1: Compute absolute difference
#         error_map = cv2.absdiff(original_gray, tampered_gray)

#         # Step 2: Multiply by 10 to enhance the visibility
#         error_map_scaled = np.clip(error_map * 10, 0, 255).astype(np.uint8)

#         # Step 3: Apply color map (e.g., JET colormap)
#         error_colored = cv2.applyColorMap(error_map_scaled, cv2.COLORMAP_JET)

#         # Step 4: Show the result
#         plt.figure(figsize=(8, 6))
#         plt.title("Error Map (×10) - Colored")
#         plt.imshow(cv2.cvtColor(error_colored, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.tight_layout()
#         plt.show()

# # Step 5: Optional - Save the colored error map
# cv2.imwrite("colored_error_map_x10.png", error_colored)


#         print(f"[✔] All comparison outputs saved to '{output_folder}'")

#         return {
#             "original": original,
#             "tampered": tampered,
#             "diff_gray": diff_gray,
#             "diff_rgb": diff_rgb,
#             "delta_e_gray": delta_e_norm,
#             "delta_e_colored": delta_e_color,
#             "highlighted_diff": highlight,
#             "ssim_thresh": ssim_thresh,
#             "final_binary": final_binary,
#             "cropped_changed_region": cropped_changes,
#             "comparison_output":comparison,
#             "colored_error_map_x10":error_colored
            
#         }

#     except Exception as e:
#         print(f"[✘] Error occurred: {str(e)}")
#         return None

# def highlight_structural_differences(original_path, tampered_path, output_folder="structure_diff_only"):
#     """Highlights only structural differences in white, everything else in black"""
#     try:
#         original = cv2.imread(original_path)
#         tampered = cv2.imread(tampered_path)

#         if original is None or tampered is None:
#             raise ValueError("Failed to load one or both images.")

#         if original.shape != tampered.shape:
#             tampered = cv2.resize(tampered, (original.shape[1], original.shape[0]))

#         # Convert to grayscale
#         original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#         tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

#         # Structural differences
#         diff = cv2.absdiff(original_gray, tampered_gray)

#         # Threshold the difference to isolate actual changes
#         _, structural_diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

#         # Output image: white = change, black = same
#         diff_output = cv2.cvtColor(structural_diff, cv2.COLOR_GRAY2BGR)  # For saving in 3 channels

#         # Save result
#         os.makedirs(output_folder, exist_ok=True)
#         output_path = os.path.join(output_folder, "structural_diff_white_only.png")
#         cv2.imwrite(output_path, diff_output)

#         print(f"[✔] Structural differences highlighted and saved to: {output_path}")
#         return diff_output

#     except Exception as e:
#         print(f"[✘] Error: {str(e)}")
#         return None
# # Example detection functions (implement according to your specific methods)
# def detect_iml_vit(original, tampered):
#     """
#     IML-VIT (Image Manipulation Localization using Vision Transformer) detection
#     Placeholder implementation - replace with actual model inference
#     """
#     try:
#         # Convert images to grayscale
#         gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#         gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        
#         # Calculate absolute difference as placeholder
#         diff = cv2.absdiff(gray_original, gray_tampered)
        
#         # Normalize to 0-1 range
#         diff_normalized = diff.astype(np.float32) / 255.0
        
#         # Apply threshold to create binary mask (placeholder)
#         _, binary_mask = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)
        
#         return binary_mask
    
#     except Exception as e:
#         print(f"Error in IML-VIT detection: {str(e)}")
#         return None

# def structural_similarity(im1, im2):
#     """
#     Compute SSIM map between two images
#     """
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(im1, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(im2, -1, window)[5:-5, 5:-5]
    
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = cv2.filter2D(im1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(im2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(im1*im2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean(), ssim_map
# # ... implement other detection functions similarly


# def detect_mvss_net(original, tampered):
#     """
#     MVSS-Net detection placeholder using skimage's SSIM
#     """
#     try:
#         # Convert to grayscale
#         gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#         gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

#         # Use skimage's SSIM with full map
#         score, diff = ssim(gray_original, gray_tampered, full=True)
#         diff = (diff * 255).astype(np.uint8)

#         # Threshold difference map
#         _, thresh = cv2.threshold(diff, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#         return thresh

#     except Exception as e:
#         print(f"Error in MVSS-Net detection: {str(e)}")
#         return None
    
# def detect_pscc_net(original, tampered):
#     """
#     PSCC-Net detection placeholder
#     Replace with actual PSCC-Net implementation
#     """
#     try:
#         # Placeholder using Canny edge differences
#         edges_original = cv2.Canny(original, 100, 200)
#         edges_tampered = cv2.Canny(tampered, 100, 200)
        
#         diff = cv2.absdiff(edges_original, edges_tampered)
#         diff_normalized = diff.astype(np.float32) / 255.0
        
#         return diff_normalized
    
#     except Exception as e:
#         print(f"Error in PSCC-Net detection: {str(e)}")
#         return None

# def detect_custom_method(original, tampered):
#     """
#     Your custom detection method
#     Implement your actual method here
#     """
#     try:
#         # Example: Simple difference with noise reduction
#         gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#         gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
        
#         diff = cv2.absdiff(gray_original, gray_tampered)
        
#         # Apply Gaussian blur to reduce noise
#         diff = cv2.GaussianBlur(diff, (5,5), 0)
        
#         # Adaptive thresholding
#         thresh = cv2.adaptiveThreshold(
#             diff, 1, 
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY, 11, 2)
        
#         return thresh
    
#     except Exception as e:
#         print(f"Error in custom method detection: {str(e)}")
#         return None


# # [Other tamper detection functions would follow...]
# def result_compare_images(original_path, tampered_path):
#     """Compares original and tampered images using AI-based tamper detection."""
#     result_paths = {}
#     try:
#         logging.info("Comparing images for tamper detection...")
#         original = cv2.imread(original_path)
#         tampered = cv2.imread(tampered_path)

#         if original is None or tampered is None:
#             raise ValueError("Failed to load images")
        
#         if original.shape != tampered.shape:
#             tampered = cv2.resize(tampered, (original.shape[1], original.shape[0]))

#         # Basic difference methods
#         gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#         gray_tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
#         diff = cv2.absdiff(gray_original, gray_tampered)
#         _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

#         # Save outputs
#         save_image(original, os.path.join(OUTPUT_FOLDER, "01_Original.png"))
#         save_image(tampered, os.path.join(OUTPUT_FOLDER, "02_Tampered.png"))
#         save_image(diff, os.path.join(OUTPUT_FOLDER, "03_Difference.png"))
#         save_image(threshold_diff, os.path.join(OUTPUT_FOLDER, "04_Threshold_Difference.png"))

#         result_paths["Original"] = os.path.join(OUTPUT_FOLDER, "01_Original.png")
#         result_paths["Tampered"] = os.path.join(OUTPUT_FOLDER, "02_Tampered.png")
#         result_paths["Diff"] = os.path.join(OUTPUT_FOLDER, "03_Difference.png")
#         result_paths["Threshold"] = os.path.join(OUTPUT_FOLDER, "04_Threshold_Difference.png")

#         # Process with advanced methods
#         models = {
#             'IML_VIT': detect_iml_vit,
#             'MVSS_Net': detect_mvss_net,
#             'PSCC_Net': detect_pscc_net,
#             'Custom': detect_custom_method
#         }

#         for model_name, detector in models.items():
#             logging.info(f"Running {model_name}...")
#             tamper_map = detector(original.copy(), tampered.copy())
#             if tamper_map is not None:
#                 output_path = os.path.join(OUTPUT_FOLDER, f"05_{model_name}.png")
#                 save_image((tamper_map * 255).astype(np.uint8), output_path)
#                 result_paths[model_name] = output_path

#         logging.info("Tamper detection completed successfully.")
#         return result_pathsa

#     except Exception as e:
#         logging.error(f"Error in result_compare_images: {str(e)}")
#         return None
