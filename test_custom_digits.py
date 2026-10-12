# Furqan Abid
# April 2026
#
# Script that tests the trained MNIST CNN model on custom handwritten
# digit images. Loads images from a specified directory, preprocesses
# each by converting to grayscale, resizing to 28x28, inverting
# intensities to match MNIST format (white on black), and applying
# normalization. Runs each image through the network and displays
# predictions in a grid plot. Optionally extracts expected labels
# from filenames to show correctness.

# import statements
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

# Import the network class from the training script
from mnist_recognition import MyNetwork


# Loads the trained MNIST model from a saved state dictionary file and
# sets it to evaluation mode so that dropout is disabled and outputs
# are deterministic.
#
# Arguments:
# model_path (str) - file path to the saved model state dictionary
#
# Returns:
# model (MyNetwork) - the trained network ready for inference
def load_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


# Preprocesses a single handwritten digit image to match MNIST format.
# Converts to grayscale, resizes to 28x28 with anti-aliasing, inverts
# intensities so digits appear white on black, applies a slight
# Gaussian blur to smooth edges, and normalizes with MNIST statistics.
# Returns both the displayable PIL image and the normalized tensor.
#
# Arguments:
# image_path (str) - file path to the handwritten digit image
#
# Returns:
# image (PIL.Image) - preprocessed grayscale image for display
# tensor (Tensor)   - normalized image tensor with batch dimension for inference
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    image = ImageOps.invert(image)

    # Apply a slight Gaussian blur to smooth the edges, similar to MNIST
    image = image.filter(ImageFilter.GaussianBlur(radius=0.3))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(image).unsqueeze(0)
    return image, tensor


# Runs the model on a list of custom digit image paths. For each
# image, preprocesses it and performs a forward pass to get the
# predicted digit. Returns the collected images and predictions
# for plotting.
#
# Arguments:
# model (MyNetwork)     - the trained network in evaluation mode
# image_paths (list)    - list of file paths to custom digit images
#
# Returns:
# images (list)      - list of preprocessed PIL images for display
# predictions (list) - list of predicted digit labels (0-9)
def evaluate_custom_digits(model, image_paths):
    images = []
    predictions = []

    with torch.no_grad():
        for image_path in image_paths:
            image, tensor = preprocess_image(image_path)
            output = model(tensor)
            pred = output.argmax(dim=1, keepdim=True).item()

            images.append(image)
            predictions.append(pred)

    return images, predictions


# Plots the custom digit images in a grid with predicted labels as
# titles. If expected labels are provided, correct predictions are
# displayed in green and incorrect ones in red. Saves the resulting
# figure to a PNG file.
#
# Arguments:
# images (list)          - list of preprocessed PIL images to display
# predictions (list)     - list of predicted digit labels
# expected_labels (list) - optional list of correct labels for comparison
def plot_custom_predictions(images, predictions, expected_labels=None):
    n = len(images)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row][col]

        if i < n:
            axes[row][col].imshow(images[i], cmap='gray')
            title = f'Pred: {predictions[i]}'
            color = 'black'
            if expected_labels is not None:
                correct = predictions[i] == expected_labels[i]
                title = f'Pred: {predictions[i]} (True: {expected_labels[i]})'
                color = 'green' if correct else 'red'
            axes[row][col].set_title(title, color=color, fontsize=11)

        axes[row][col].axis('off')

    plt.suptitle('Custom Handwritten Digits - Network Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig('custom_digit_predictions.png')
    plt.show()
    print("Custom predictions plot saved to custom_digit_predictions.png")


# Main function that orchestrates testing on custom handwritten digit
# images. Reads all image files from the specified directory, attempts
# to extract expected labels from filenames, loads the trained model,
# evaluates each image, and generates a prediction grid plot.
#
# Arguments:
# argv (list) - command line arguments: [image_directory]
def main(argv):
    image_dir = 'digits'

    if len(argv) > 1:
        image_dir = argv[1]

    if not os.path.isdir(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        print("Usage: python test_custom_digits.py [image_directory]")
        return

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print(f"No image files found in directory '{image_dir}'.")
        return

    image_paths = [os.path.join(image_dir, f) for f in image_files]

    # Try to extract expected labels from filenames
    # Looks for a digit in the filename (e.g., digit_0.png -> 0, digit_3.png -> 3)
    expected_labels = []
    for f in image_files:
        digits_in_name = [c for c in os.path.splitext(f)[0] if c.isdigit()]
        if digits_in_name:
            expected_labels.append(int(digits_in_name[-1]))
        else:
            expected_labels.append(None)
            break

    print(f"Found {len(image_paths)} images in '{image_dir}'.\n")

    model = load_model('mnist_model.pth')
    images, predictions = evaluate_custom_digits(model, image_paths)
    plot_custom_predictions(images, predictions, expected_labels)


if __name__ == "__main__":
    main(sys.argv)