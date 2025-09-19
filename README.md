# CNN for Image Classification

This project demonstrates how to build and train a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The model is trained on the CIFAR-10 dataset.

## Features

- **CNN Model**: A sequential CNN model with multiple convolutional, batch normalization, and max-pooling layers.
- **Data Augmentation**: Uses `ImageDataGenerator` to perform real-time data augmentation, improving model generalization.
- **Callbacks**: Implements `ModelCheckpoint` to save the best model and `EarlyStopping` to prevent overfitting.
- **Web UI**: A simple Flask application to upload an image and get a prediction.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- Flask

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Train the model**:
   ```bash
   python main.py
   ```
   This will train the model and save the best weights to `best_model.h5`.

2. **Run the web application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000` to use the application.
# CNN-for-Image-Classification
