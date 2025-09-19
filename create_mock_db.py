import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import array_to_img

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def create_mock_database(base_dir='cifar10_mock_db', num_images_per_class=10):
    """Creates a mock database by saving a subset of CIFAR-10 images to disk."""
    (x_train, y_train), _ = cifar10.load_data()

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Get indices for the current class
        class_indices = np.where(y_train == i)[0]
        # Select a subset of images
        images_to_save = class_indices[:num_images_per_class]

        for j, index in enumerate(images_to_save):
            img = array_to_img(x_train[index])
            img.save(os.path.join(class_dir, f'{class_name}_{j}.png'))

    print(f"Mock database created at '{base_dir}' with {num_images_per_class} images per class.")

if __name__ == '__main__':
    create_mock_database()
