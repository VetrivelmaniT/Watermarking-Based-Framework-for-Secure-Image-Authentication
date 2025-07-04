import logging
from watermark_comparison import WatermarkComparison
from aigc_editing_methods import AIGCEditingMethods 
from tamper_localization import TamperLocalization
from copyright_protection import CopyrightProtection

def run_watermark_analysis():
    """Runs watermark comparison, AIGC editing methods, tamper localization, and copyright protection analysis."""
    try:
        logging.info("Starting watermark analysis...")
        comparator = WatermarkComparison()
        logging.info(f"Fidelity Scores: {comparator.compare_fidelity()}")
        logging.info(f"Perceptual Quality Scores: {comparator.evaluate_perceptual_quality()}")

        editor = AIGCEditingMethods()
        editor.prepare_dataset()
        logging.info(f"Editing Method Performance: {editor.evaluate_methods()}")

        localizer = TamperLocalization()
        logging.info(f"Tamper Localization Accuracy: {localizer.analyze_tampering()}")

        protector = CopyrightProtection()
        logging.info(protector.recover_copyright())
    except Exception as e:
        logging.error(f"Error in watermark analysis: {e}")