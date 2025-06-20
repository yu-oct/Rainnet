import h5py
import numpy as np
import os

# Define the target crop size (128 × 128 pixels)
crop_size = 128
half_crop = crop_size // 2

# Define the center coordinates for cropping (based on full radar image size)
center_x, center_y = 2400, 2200

# List of 4 sequential radar files spaced at 5-minute intervals
file_paths = [
    "D:/Postgraduate/Data Science Dissertation/rainnet/composite_dmax_20250615_1750-hd5",
    "D:/Postgraduate/Data Science Dissertation/rainnet/composite_dmax_20250615_1755-hd5",
    "D:/Postgraduate/Data Science Dissertation/rainnet/composite_dmax_20250615_1800-hd5",
    "D:/Postgraduate/Data Science Dissertation/rainnet/composite_dmax_20250615_1805-hd5"
]

frames = []  # List to store cropped and log-transformed frames

# Loop through each radar file and extract a cropped and transformed frame
for path in file_paths:
    with h5py.File(path, "r") as f:
        # Load the full radar image (typically ~4800x4400 pixels)
        full_frame = f["dataset1/data1/data"][()]
        
        # Spatial cropping centered at (center_x, center_y)
        crop = full_frame[
            center_x - half_crop : center_x + half_crop,
            center_y - half_crop : center_y + half_crop
        ]

        # Apply logarithmic transformation to match RainNet's expected input
        # Formula: log(mm + 0.01) to avoid log(0)
        crop_log = np.log(crop + 0.01)

        # Append the processed frame to the list
        frames.append(crop_log)

# Stack the 4 frames along the last axis to form a tensor of shape (128, 128, 4)
X_input = np.stack(frames, axis=-1)

# Add batch dimension at the front: final shape = (1, 128, 128, 4)
X_input = X_input[np.newaxis, ...]

# Save the preprocessed input array to a .npy file for later model inference
np.save(
    "D:/Postgraduate/Data Science Dissertation/rainnet/X_input_cropped.npy",
    X_input
)


