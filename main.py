import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.metrics import peak_signal_noise_ratio as psnr

# Model definition
def create_denoising_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(None, None, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    return model

# Data loading function
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = img.astype(np.float32) / 255.0
            images.append(img)
            filenames.append(filename)
        else:
            print(f"Warning: Couldn't read image {img_path}")
    return np.array(images), filenames

# Training function
def train_model(model, train_low_dir, train_high_dir):
    low_images, _ = load_images_from_folder(train_low_dir)
    high_images, _ = load_images_from_folder(train_high_dir)
    model.compile(optimizer='adam', loss='mse')
    model.fit(low_images, high_images, epochs=50, batch_size=16, validation_split=0.1)
    model.save('denoising_model.h5')

# Evaluation function
def evaluate_model(model, low_images, high_images):
    predicted_images = model.predict(low_images)
    psnr_values = [psnr(high_images[i], predicted_images[i]) for i in range(len(high_images))]
    return np.mean(psnr_values)

# Denoising and saving results function
def denoise_and_save(model, test_low_dir, test_pred_dir):
    test_images, filenames = load_images_from_folder(test_low_dir)
    predicted_images = model.predict(test_images)
    for i, img in enumerate(predicted_images):
        save_path = os.path.join(test_pred_dir, filenames[i])
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))

# Main execution
if __name__ == "__main__":
    # Create the model
    model = create_denoising_model()
    
    # Define paths
    train_low_dir = './train/low/'
    train_high_dir = './train/high/'
    test_low_dir = './test/low/'
    test_pred_dir = './test/predicted/'

    # Train the model if needed
    if not os.path.exists('denoising_model.h5'):
        train_model(model, train_low_dir, train_high_dir)
    
    # Load the trained model
    model = tf.keras.models.load_model('denoising_model.h5')

    # Ensure the output directory exists
    if not os.path.exists(test_pred_dir):
        os.makedirs(test_pred_dir)

    # Denoise the test images and save the results
    denoise_and_save(model, test_low_dir, test_pred_dir)
