import os
import shutil
import random

# Define directories for SPLIT2 and SPLIT3
data_path = 'data/'
train_dir = os.path.join(data_path, 'train')
val_dir = os.path.join(data_path, 'val')
split3_train_dir = os.path.join(data_path, 'SPLIT3/TRAIN')
split3_val_dir = os.path.join(data_path, 'SPLIT3/VAL')

# Create SPLIT3 directories if they don't exist
os.makedirs(split3_train_dir, exist_ok=True)
os.makedirs(split3_val_dir, exist_ok=True)

for category in os.listdir(train_dir):
    src_category_dir = os.path.join(train_dir, category)
    dst_category_dir = os.path.join(split3_train_dir, category)
    os.makedirs(dst_category_dir, exist_ok=True)
    for file in os.listdir(src_category_dir):
        shutil.copy(os.path.join(src_category_dir, file), os.path.join(dst_category_dir, file))

for category in os.listdir(val_dir):
    src_category_dir = os.path.join(val_dir, category)
    dst_category_dir = os.path.join(split3_train_dir, category)
    os.makedirs(dst_category_dir, exist_ok=True)
    for file in os.listdir(src_category_dir):
        shutil.copy(os.path.join(src_category_dir, file), os.path.join(dst_category_dir, file))