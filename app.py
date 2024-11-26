# Import required modules
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define the image captioning endpoint
@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    # Check if an image is part of the POST request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Get the image file from the request
    image_file = request.files['image']
    
    # Open the image
    image = Image.open(image_file.stream)
    
    # Preprocess the image and generate caption
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Return the caption in JSON format
    return jsonify({'caption': caption})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
