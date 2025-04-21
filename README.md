# Hyperglycemia-related Retinal Disorder Prediction

This project implements a deep learning-based system for predicting hyperglycemia-related retinal disorders using retinal images. The system uses a Convolutional Neural Network (CNN) to analyze retinal images and classify them into different categories of retinal disorders.

## Features

- Web-based interface for easy image upload and prediction
- Real-time prediction of retinal disorders
- Support for multiple image formats (PNG, JPG, JPEG)
- Confidence score for predictions
- User-friendly interface with error handling

## Project Structure

```
├── app.py                 # Flask web application
├── train.py              # Model training script
├── gui.py               # GUI implementation
├── eye_disease_cnn_model.h5  # Trained model file
├── dataset/             # Dataset directory
├── static/             # Static files (CSS, JS, uploads)
└── templates/          # HTML templates
```

## Requirements

- Python 3.x
- TensorFlow
- Flask
- NumPy
- Pillow
- Werkzeug

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Hyperglycemia-releted-retinal-disorder-prediction.git
cd Hyperglycemia-releted-retinal-disorder-prediction
```

2. Install the required packages:
```bash
pip install tensorflow flask numpy pillow werkzeug
```

## Usage

### Training the Model

To train the model on your dataset:

```bash
python train.py
```

The training script will:
- Load and preprocess the images
- Train the CNN model
- Save the trained model as 'eye_disease_cnn_model.h5'

### Running the Web Application

To start the web application:

```bash
python app.py
```

The application will automatically open in your default web browser at `http://127.0.0.1:5000/`.

### Using the GUI

To use the graphical user interface:

```bash
python gui.py
```

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers with dropout for regularization
- Softmax output layer for classification

## Dataset

The model is trained on retinal images stored in the `dataset/test` directory. The dataset should be organized with subdirectories for each class of retinal disorder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped with the project
- Special thanks to the open-source community for the tools and libraries used in this project 