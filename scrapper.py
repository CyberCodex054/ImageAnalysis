import os
import cv2
import pytesseract
import json
import datetime
from transformers import pipeline
from PIL import Image
import re
import uuid

# Path to the folder containing memes (Update this to your meme folder)
meme_folder = os.path.expanduser(r'C:\Users\Medwe\OneDrive\Desktop\project')

# Path to save processed memes (Saving in Downloads)
meme_save_path = os.path.expanduser(r'C:\Users\Medwe\Downloads')

# Load the image captioning model


caption_generator = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


def sanitize_filename(filename):
    """Sanitize filenames to remove invalid characters and limit length."""
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return safe[:50]  # Limit filename length


def check_folders():
    """Check if meme folder exists, otherwise print error."""
    if not os.path.exists(meme_folder):
        raise FileNotFoundError(f"Specified meme folder does not exist: {meme_folder}")
    if not os.path.exists(meme_save_path):
        os.makedirs(meme_save_path)  # Create save folder if it doesn't exist


def analyze_meme(image_path):
    """Analyze meme image to extract text and faces, and generate a caption."""
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert BGR to RGB for OCR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(image_rgb)

        # Detect faces using OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file missing: {cascade_path}")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise ValueError("Haar cascade file could not be loaded.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Check for potential meme patterns
        is_potential_meme = False
        if len(text.strip()) > 20:
            is_potential_meme = True
        if len(faces) > 0:
            is_potential_meme = True

        # Generate AI caption using the HuggingFace model
        try:
            pil_image = Image.open(image_path).convert("RGB")
            caption = caption_generator(pil_image)[0]['generated_text']
        except Exception as ce:
            print(f"Caption generation failed for {image_path}: {ce}")
            caption = ""

        # Prepare JSON output
        output = {
            'filename': os.path.basename(image_path),
            'text': text,
            'caption': caption,
            'processed_time': datetime.datetime.now().isoformat()
        }

        return output, is_potential_meme
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, False


def main():
    """Main function to process memes and save results."""
    try:
        check_folders()  # Ensure folders are valid and exist

        # Process each meme in the folder
        meme_results = []
        for filename in os.listdir(meme_folder):
            filename_lower = filename.lower()
            if filename_lower.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(meme_folder, filename)
                meme_result, is_meme = analyze_meme(image_path)
                if meme_result:
                    meme_results.append(meme_result)
                    if is_meme:
                        # Sanitize caption to make it a valid filename
                        caption = sanitize_filename(meme_result['caption'])
                        if not caption.strip():
                            caption = "untitled_" + str(uuid.uuid4())[:8]
                        save_path = os.path.join(meme_save_path, f"{caption}.jpg")
                        image_to_save = cv2.imread(image_path)
                        if image_to_save is not None:
                            cv2.imwrite(save_path, image_to_save)
                        else:
                            print(f"Failed to load image for saving: {image_path}")

        # Save the meme results to a JSON file
        result_file = os.path.expanduser('~/Downloads/meme_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(meme_results, f, indent=2, ensure_ascii=False)

        print(f"Meme analysis completed. Results saved to {result_file}")

    except Exception as e:
        print(f"Error during meme analysis: {e}")


if __name__ == '__main__':
    main()