import logging
from modules.image_processing import process_image
from modules.tamper_detection import compare_images, result_compare_images
from modules.watermark_analysis import run_watermark_analysis

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Main function to execute tamper detection and watermark analysis."""
    original_path = "t4 (1).jpg"   # Change to actual original image path
    tampered_path = "t4 (2).jpg"  # Change to actual tampered image path

    logging.info("Starting Image Processing Pipeline...")

    # Step 1: Process Image (Watermarking and Enhancement)
    process_image(original_path)
    
    # Step 2: Compare Images (Tamper Detection)
    result_compare_images(original_path, tampered_path)
    compare_images(original_path, tampered_path)

    # Step 3: Perform Watermark Analysis
    run_watermark_analysis()

    logging.info("Process completed successfully!")

    
    
if __name__ == "__main__":
    main()