from transformers import pipeline
from PIL import Image
import os
import json
import torch

# Load classifier with GPU if available
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)

# Image path (single file or folder)
image_path = os.path.expanduser(image_path)

# Store results
classification_results = {}

# Get image files
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

# Classify images
for image_file in image_files:
    try:
        image = Image.open(image_file).convert("RGB")
        results = classifier(image)
        top5 = [
            {"label": result["label"], "score": result["score"]}
            for result in results[:5]
        ]
        classification_results[os.path.basename(image_file)] = top5
        print(f"\nProcessed: {image_file}")
        print(json.dumps(top5, indent=2))
    except Exception as e:
        error_msg = {
            "error": str(e),
        }
        classification_results[os.path.basename(image_file)] = error_msg
        print(f"\nError processing {image_file}: {e}")

# Save results
output_file = "image_classification_results.json"
with open(output_file, "w") as f:
    json.dump(classification_results, f, indent=2)

print(f"\n✅ Classification complete. Results saved to {output_file}")
