import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_data():
    """Loads and preprocesses the CIFAR-10 dataset."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Creates and compiles the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Trains the model with data augmentation and callbacks."""
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                        epochs=50, 
                        validation_data=(x_test, y_test), 
                        callbacks=[checkpoint, early_stopping])
    return history

def evaluate_model(model, x_test, y_test):
    """Evaluates the model on the test set."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")

def plot_predictions(model, x_test):
    """Plots predictions for a few test images."""
    predictions = model.predict(x_test[:10])

    plt.figure(figsize=(12, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(x_test[i])
        plt.title(CLASS_NAMES[np.argmax(predictions[i])])
        plt.axis('off')
    plt.suptitle("CIFAR-10 Predictions (First 10 Samples)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = create_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    plot_predictions(model, x_test)
