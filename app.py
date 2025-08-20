from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = load_model("model_new_lenet_88_acc.keras")  
image_size = (224, 224)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Normalize pixel values
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

class_labels = ['lsb', 'non_stego']  # Use your correct order

@app.route('/detect', methods=['POST'])
def detect():
    class_labels = ['lsb', 'non_stego']  # Make sure it's inside the function

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(filepath)
    
    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    label = class_labels[predicted_class]  # will be 'lsb' or 'non_stego'
    
    # Optional: convert to readable text for frontend
    readable_result = "Stego Image (LSB)" if label == 'lsb' else "Non-Stego Image"
    
    return jsonify({'result': readable_result, 'image_url': filepath})



# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'})
    
#     image_file = request.files['image']
#     if image_file.filename == '':
#         return jsonify({'error': 'No selected image'})
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
#     image_file.save(filepath)
    
#     img_array = preprocess_image(filepath)
#     prediction = model.predict(img_array)
#     result = "Stego Image" if np.argmax(prediction) == 1 else "Non-Steg Image"
    
#     return jsonify({'result': result, 'image_url': filepath})

from flask import send_from_directory

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True)

