# 🌽 Corn Leaf Doctor – Deep Learning Based Disease Detection

## Project Overview

Corn Leaf Doctor is a deep learning application designed to identify whether a corn leaf is healthy or affected by disease using image classification.

The application uses a trained MobileNetV2-based deep learning model to analyze uploaded images or webcam captures and predict the condition of the leaf.

## Features

- Image-based corn leaf disease detection
- Real-time webcam detection
- Healthy and diseased classification
- Prediction confidence score
- Low-confidence prediction handling
- Interactive Streamlit web interface

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Streamlit
- PIL (Pillow)
- Jupyter Notebook

## Model Details

- Architecture: MobileNetV2
- Input image size: 224 × 224
- Classification: Binary classification
- Classes: Healthy and Diseased
- Confidence threshold: 70% in the Streamlit application

The application identifies disease symptoms associated with Northern Leaf Blight in the dataset.


## How It Works

1. Upload a corn leaf image or capture one using the webcam.
2. The image is resized to 224 × 224 pixels.
3. MobileNetV2 preprocessing is applied.
4. The trained model predicts the leaf condition.
5. The application displays the classification and confidence score.

## Disclaimer

This project is developed for educational and demonstration purposes. Predictions are not a substitute for professional agricultural diagnosis.

## Author

**MOHANAKRISHNAN.M**

GitHub: https://github.com/mhn-alt

