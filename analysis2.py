import json
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import warnings

# Optional: Suppress symlink warning (Windows dev mode issue)
warnings.filterwarnings("ignore", message=".*symlinks on Windows.*")

# Use a pre-trained model with fully initialized weights
model_id = "google/vit-base-patch16-224"  # Fully trained on ImageNet

# Image to classify
image_path =image_path
output_file = "image_classification_results.json"

# Load model and image processor
model = AutoModelForImageClassification.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

# Load image
image = Image.open(image_path).convert("RGB")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create pipeline
classifier = pipeline("image-classification", model=model, image_processor=processor, device=0 if device.type == "cuda" else -1)

# Run classification
results = classifier(image, top_k=3)  # Top 3 predictions

# Format results
output = {
    "file": image_path,
    "top_predictions": results
}

# Save results
with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print("✅ Classification complete. Results saved to", output_file)
