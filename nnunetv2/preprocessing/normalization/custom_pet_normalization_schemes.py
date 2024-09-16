from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization
import numpy as np

class SUVClipNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "SUVClipNormalization requires intensity properties"
        lower_bound = self.intensityproperties['minSUV']
        upper_bound = self.intensityproperties['maxSUV']

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        return image

class SUVPercentClipNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "SUVPercentClipNormalization requires intensity properties"
        lower_bound = 0
        upper_bound = np.max(image) * (self.intensityproperties['percentSUV']/100)

        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)
        return image
