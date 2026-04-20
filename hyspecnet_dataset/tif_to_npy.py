import glob
import multiprocessing
import numpy as np
import rasterio

in_directory = "/media/SSD2/piccinini/hyspecnet-11k/hyspecnet-11k/patches/"


invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
valid_channels_ids = [c+1 for c in range(224) if c not in invalid_channels]

minimum_value = 0
maximum_value = 10000


in_patches = glob.glob(f"{in_directory}**/**/*SPECTRAL_IMAGE.TIF")


def convert(patch_path):
    # load patch
    dataset = rasterio.open(patch_path)
    # remove nodata channels
    src = dataset.read(valid_channels_ids)
    # clip data to remove uncertainties
    clipped = np.clip(src, a_min=minimum_value, a_max=maximum_value)
    # min-max normalization
    out_data = (clipped - minimum_value) / (maximum_value - minimum_value)
    out_data = out_data.astype(np.float32)
    # save npy
    out_path = patch_path.replace("SPECTRAL_IMAGE", "DATA").replace("TIF", "npy")
    np.save(out_path, out_data)
    
with multiprocessing.Pool(64) as pool:
    pool.map(convert, in_patches)