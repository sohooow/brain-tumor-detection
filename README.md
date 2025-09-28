# Brain Tumor MRI Detection

## Overview
This project provides an end-to-end solution for automated brain tumor detection and classification using deep learning techniques. It includes a Jupyter notebook for model development and a modern Flask web application for real-time MRI image analysis.

<img width="1470" height="919" alt="Screenshot 2025-09-28 at 16 25 46" src="https://github.com/user-attachments/assets/192b3b61-efb4-4307-ac60-5c755a4a2324" />

<img width="1470" height="919" alt="Screenshot 2025-09-28 at 16 10 24" src="https://github.com/user-attachments/assets/e001c7d7-61d1-4150-a73d-81d5cd2142a3" />

<img width="1470" height="919" alt="Screenshot 2025-09-28 at 16 09 23" src="https://github.com/user-attachments/assets/79f5958d-b82c-40f3-9d54-b133c7237f1b" />

## Features
- Convolutional Neural Network (CNN) for multi-class brain tumor classification
- Preprocessing pipeline: cropping, resizing, normalization
- Data augmentation for robust training
- Model evaluation with accuracy, precision, recall, confusion matrix, and ROC curves
- Flask web app for uploading MRI images and visualizing predictions
- Responsive, modern UI with glassmorphism and smooth transitions

## Project Structure
```
Brain Tumor MRI Detection/
│
├── main.py                  # Flask backend for web app
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── model/
│   ├── BrainTumorDetection.ipynb  # Jupyter notebook for model training
│   └── model.h5             # Trained Keras model
├── templates/
│   └── index.html           # Web app UI template
├── uploads/                 # Uploaded MRI images (runtime)
└── sample MRI Images/       # Example MRI images
```

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sohooow/brain-tumor-detection.git
cd "Brain Tumor MRI Detection"
```

### 2. Set Up the Python Environment
It is recommended to use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Training (Optional)
- Open `model/BrainTumorDetection.ipynb` in Jupyter or VS Code.
- Follow the notebook to preprocess data, train, and export the model as `model.h5`.
- You can use your own MRI dataset or the provided sample images.

### 5. Run the Web Application
```bash
python main.py
```
- The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)
- Upload an MRI image to get a prediction and confidence score.

## Model Details
- **Architecture:** Custom CNN with multiple convolutional, pooling, dropout, and dense layers
- **Input Size:** 225x225x3 (cropped and normalized MRI images)
- **Classes:** No Tumor, Pituitary Tumor, Meningioma Tumor, Glioma Tumor
- **Metrics:** Accuracy, Precision, Recall, Confusion Matrix, ROC Curve

## Preprocessing Pipeline
- **Cropping:** Removes empty space around the brain region
- **Resizing:** All images resized to 225x225 pixels
- **Normalization:** Pixel values scaled to [0, 1] using OpenCV normalization
- **Augmentation:** Rotation, shear, zoom, and horizontal flip during training

## Web Application
- **Framework:** Flask (Python)
- **Frontend:** HTML5, CSS3 (glassmorphism), Bootstrap 5
- **Features:**
  - Upload MRI images (PNG, JPG, etc.)
  - Real-time prediction with confidence score
  - Result color-coded (green for no tumor, red for tumor detected)
  - Responsive and modern design
  - Option to analyze multiple images in one session

## Example Usage
1. Start the Flask app: `python main.py`
2. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)
3. Upload an MRI image and view the prediction result

## Notes
- This tool is for research and educational purposes only. It is not intended for clinical use.
- Always consult a medical professional for diagnosis and treatment decisions.
