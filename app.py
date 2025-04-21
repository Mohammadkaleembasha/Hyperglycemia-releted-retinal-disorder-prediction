from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import uuid
from werkzeug.utils import secure_filename
import webbrowser
from threading import Timer

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained model
try:
    model_path = 'eye_disease_cnn_model.h5'
    model = load_model(model_path)
    # Set image dimensions and class names
    img_height, img_width = 64, 64
    class_names = sorted(os.listdir('dataset/test'))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict the class of an image
def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        predicted_label = class_names[predicted_class]
        return predicted_label, confidence, None
    except Exception as e:
        return None, None, str(e)

# Function to open browser
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

# Routes
@app.route('/')
def index():
    if model is None:
        flash('Model not loaded. Please check the model file.', 'error')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded. Please check the model file.', 'error')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload JPG, JPEG or PNG files.', 'error')
        return redirect(url_for('index'))

    try:
        # Generate secure filename with UUID to prevent duplicates
        filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{filename}"
        
        # Save the file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Make prediction using the full path
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        predicted_label, confidence, error = predict_image(filepath)
        
        if error:
            flash(f'Error during prediction: {error}', 'error')
            return redirect(url_for('index'))
        
        return render_template(
            'result.html',
            image_path=filename,  # Just pass the filename, not the full path
            prediction=predicted_label,
            confidence=f"{confidence:.2f}%"
        )

    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))

# Run Flask application
if __name__ == '__main__':
    # Open browser after 1.5 seconds
    Timer(1.5, open_browser).start()
    app.run(debug=True)
