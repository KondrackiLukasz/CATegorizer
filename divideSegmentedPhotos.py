import os
import glob
import shutil
from random import shuffle

# path to your dataset
data_path = "C:/Users/lukasz/PycharmProjects/CATegorizer/data/segmented/"

# create train, validation and test folders if not existing
for set_name in ["train", "val", "test"]:
    if not os.path.exists(f'{data_path}/{set_name}'):
        os.makedirs(f'{data_path}/{set_name}')

# get the class labels
labels = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]

# loop through each label
for label in labels:
    # get a list of image file paths
    image_files = glob.glob(f"{data_path}/{label}/*.jpg") # adjust the file type if needed

    # shuffle the data order
    shuffle(image_files)

    # calculate split indices
    train_idx = int(len(image_files) * 0.8)
    val_idx = int(len(image_files) * 0.9)

    # split the data into train, validation and test sets
    train_files = image_files[:train_idx]
    val_files = image_files[train_idx:val_idx]
    test_files = image_files[val_idx:]

    # copy files into respective directories
    for set_name, file_set in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        if not os.path.exists(f'{data_path}/{set_name}/{label}'):
            os.makedirs(f'{data_path}/{set_name}/{label}')
        for file in file_set:
            shutil.move(file, f'{data_path}/{set_name}/{label}/')
