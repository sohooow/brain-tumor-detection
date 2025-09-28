from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Set logging level to reduce verbosity
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Load the trained model
model = load_model('model/model.h5')

# Class labels (matching the order from the notebook)
class_labels = ['no_tumor','pituitary_tumor', 'meningioma_tumor','glioma_tumor']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to crop an image (same as in notebook)
def crop_image(image):
    # Convert the image to grayscale
    if len(image.shape) == 3:  # Check if the image is in BGR format
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Image is already in grayscale
        gray = image.copy()

    # Threshold the image
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Crop the image to the largest contour
    cropped_image = image[
        cv2.boundingRect(largest_contour)[1] : cv2.boundingRect(largest_contour)[1] + cv2.boundingRect(largest_contour)[3],
        cv2.boundingRect(largest_contour)[0] : cv2.boundingRect(largest_contour)[0] + cv2.boundingRect(largest_contour)[2]
    ]

    return cropped_image

# Helper function to predict tumor type
def predict_tumor(image_path):
    try:
        IMAGE_SIZE = 225  # Updated to match the trained model size
        
        # Load image using OpenCV (same as notebook)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return "Error: Could not load image", 0.0
        
        # Apply cropping (same preprocessing as notebook)
        cropped_img = crop_image(img)
        
        # Resize the image
        resized_img = cv2.resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Normalize the image (same as notebook)
        normalized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Add batch dimension
        img_array = np.expand_dims(normalized_img, axis=0)

        # Make prediction (suppress verbose output)
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]
        
        print(f"Prediction result: {class_labels[predicted_class_index]}, Confidence: {confidence_score}")

        if class_labels[predicted_class_index] == 'no_tumor':
            return "No Tumor Detected", confidence_score
        else:
            tumor_type = class_labels[predicted_class_index].replace('_', ' ').title()
            return f"Tumor Detected: {tumor_type}", confidence_score
    
    except Exception as e:
        print(f"Error in predict_tumor: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 0.0

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("POST request received")
        # Handle file upload
        file = request.files.get('file')
        print(f"File received: {file}")
        
        if file and file.filename != '':
            print(f"File details - Name: {file.filename}, Content-Type: {file.content_type}")
            
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving file to: {file_location}")
            file.save(file_location)
            
            # Verify file was saved
            if os.path.exists(file_location):
                print(f"File saved successfully. Size: {os.path.getsize(file_location)} bytes")
            else:
                print("Error: File was not saved!")
                return render_template('index.html', result="Error: File upload failed", confidence="0%")

            # Predict the tumor
            print("Starting prediction...")
            result, confidence = predict_tumor(file_location)
            print(f"Prediction result: {result}, Confidence: {confidence}")

            # Return result along with image path for display
            return render_template('index.html', 
                                 result=result, 
                                 confidence=f"{confidence*100:.2f}%", 
                                 file_path=f'/uploads/{file.filename}')
        else:
            print("No file received or empty filename")
            return render_template('index.html', result="Error: No file selected", confidence="0%")

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Brain Tumor Detection Server...")
    print("Visit: http://127.0.0.1:5000")
    app.run(debug=False, host='127.0.0.1', port=5000)

    