from ibsn_model.ibsn import IBSN
from ibsn_model.mask_extractor import MaskExtractor

class TamperTrace:
    def __init__(self):
        self.ibsn = IBSN()
        self.mask_extractor = MaskExtractor()

    def detect_tampering(self, image):
        mask = self.mask_extractor(image)
        return mask
