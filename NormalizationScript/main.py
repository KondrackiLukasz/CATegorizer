import os
import numpy as np
from PIL import Image
PATH = "C:/Users/user/Desktop/szum_dataset/DatasetImages/"
def load_dataset_and_normalize():
    ret = []
    categories = {}

    i = 0
    for directory in os.listdir(PATH):
        categories[directory]=i
        for image_path in os.listdir(PATH + directory):
            ret.append([np.array(normalize(directory+"/"+image_path)),categories[directory]])
        i +=1
    return ret

def normalize(filePath):

            # open the image using PIL
            img = Image.open(PATH+filePath)

            # resize the image to 256x256
            img_resized = img.resize((256, 256))

            # convert the resized image to a numpy array
            img_arr = np.array(img_resized)

            # scale the pixel values between 0 and 1
            img_scaled = img_arr.astype(np.float32) / 255.0

            return img_scaled


dataset = load_dataset_and_normalize()
print("DANE J.A.")