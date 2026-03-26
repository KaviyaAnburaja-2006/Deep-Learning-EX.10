import os
import argparse
import pyttsx3
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    parser = argparse.ArgumentParser(description="Generate Image Caption")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--voice", action="store_true", help="Enable voice output")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path {args.image} does not exist.")
        return

    print("Loading HuggingFace Image Captioning Model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model loaded successfully!")

    print(f"Generating caption for: {args.image} ...")
    try:
        raw_image = Image.open(args.image).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).capitalize()

        
        print(f"\nCaption: {caption}")

        if args.voice:
            engine = pyttsx3.init()
            engine.say(caption)
            engine.runAndWait()
            
    except Exception as e:
        print(f"Error generating caption: {e}")

if __name__ == "__main__":
    main()
