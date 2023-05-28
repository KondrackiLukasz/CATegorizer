import os
import shutil
import numpy as np

# List of breeds (classes)
breeds = [
    "BritishShorthair",
    "Havanese",
    "Ragdoll",
    "RussianBlue",
    "Samoyed",
    "ShibaInu",
]

# The proportion of the whole dataset to include in the test split
test_split = 0.1

# The proportion of the whole dataset to include in the validation split
val_split = 0.1

# Base directories
base_images_dir = "images/"
base_output_dir = "data/"

# Loop through each breed
for breed in breeds:
    # Directory containing images of the breed
    breed_dir = os.path.join(base_images_dir, breed)

    # List of all files for the breed
    files = os.listdir(breed_dir)

    # Shuffle the files
    np.random.shuffle(files)

    # Indices for train/val/test splits
    val_split_idx = int(len(files) * (1 - (val_split + test_split)))
    test_split_idx = int(len(files) * (1 - test_split))

    # Files for train/val/test splits
    train_files = files[:val_split_idx]
    val_files = files[val_split_idx:test_split_idx]
    test_files = files[test_split_idx:]

    # Function to move files to their new directories
    def move_files(files, split):
        for file in files:
            src_path = os.path.join(breed_dir, file)
            dst_dir = os.path.join(base_output_dir, split, breed)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, file)
            shutil.move(src_path, dst_path)

    # Move files to their new directories
    move_files(train_files, "train")
    move_files(val_files, "val")
    move_files(test_files, "test")
