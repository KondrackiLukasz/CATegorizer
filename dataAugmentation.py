import os
import glob
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Replace the path with the actual path to your data folder
data_path = 'data/raw/'

# Get a list of all the breed folders
breed_folders = glob.glob(os.path.join(data_path, '*'))

# Create an ImageDataGenerator instance with the desired augmentations
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect')

# Generate and display augmented images
import matplotlib.pyplot as plt

# Configure the display settings
num_pairs = 5
plt.figure(figsize=(12, 6 * num_pairs))
plt.subplots_adjust(hspace=0.3)

for i in range(num_pairs):
    # Choose a random breed folder and a random image from that folder
    random_breed = random.choice(breed_folders)
    random_image_path = random.choice(glob.glob(os.path.join(random_breed, '*.jpg')))

    # Load the original image
    original_img = load_img(random_image_path)

    # Convert the original image to an array and reshape it
    image_array = img_to_array(original_img)
    image_array = image_array.reshape((1,) + image_array.shape)

    # Display the original image
    plt.subplot(num_pairs, 2, 2 * i + 1)
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')

    # Display an example of an augmented image
    for batch in datagen.flow(image_array, batch_size=1):
        plt.subplot(num_pairs, 2, 2 * i + 2)
        plt.title('Augmented Image')
        plt.imshow(array_to_img(batch[0]))
        plt.axis('off')
        break

plt.show()
