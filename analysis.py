from transformers import pipeline, ViTImageProcessor
from PIL import Image
import os
import json

# Load a pre-trained image classification pipeline
classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=-1)

# Path to a folder OR single image file to classify
image_path = os.path.expanduser(image_path)  # Update this to your actual path

# Store results
classification_results = {}

# Handle single file or folder
if os.path.isfile(image_path):
    image_files = [image_path]
elif os.path.isdir(image_path):
    image_files = [
        os.path.join(image_path, f)
        for f in os.listdir(image_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
else:
    raise FileNotFoundError(f"Path not found: {image_path}")

# Go through each image and classify it
for image_file in image_files:
    try:
        image = Image.open(image_file).convert("RGB")
        results = classifier(image)
        classification_results[os.path.basename(image_file)] = results[0]  # Top prediction
    except Exception as e:
        classification_results[os.path.basename(image_file)] = {"error": str(e)}

# Save the results as a JSON file
output_file = "image_classification_results.json"
with open(output_file, "w") as f:
    json.dump(classification_results, f, indent=2)

print(f"✅ Classification complete. Results saved to {output_file}")
