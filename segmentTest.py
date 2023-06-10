import os
from PIL import Image
from lang_sam import LangSAM


# Load the model
model = LangSAM()

# Input directory
input_dir = 'data/raw//'

# Output directory
output_dir = 'data/segmented/ShibaInu/'


# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Text prompt
text_prompt = "dog"
# Iterate over images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(input_dir, filename)
        img_pil = Image.open(image_path).convert("RGB")

        # Generate masks, boxes, phrases, logits
        masks, boxes, phrases, logits = model.predict(img_pil, text_prompt)

        # Assuming that the masks are binary masks (True for the object of interest, False otherwise),
        # we convert them to the right format and save the segmented images
        for i, mask in enumerate(masks):
            mask_image_pil = img_pil.copy()
            mask_image_pil.putalpha(mask.astype("uint8") * 255)  # Convert binary mask to alpha layer

            # Save the mask image
            mask_image_pil.save(os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_segmented_{i}.png'))

        break
