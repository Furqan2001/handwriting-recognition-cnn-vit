# Furqan Abid
# April 2026
#
# Script that examines a trained MNIST CNN network by analyzing its
# internal structure and first convolutional layer. Prints the model
# architecture, visualizes the 10 learned 5x5 filters from conv1 in
# a grid, and applies each filter to the first training image using
# OpenCV's filter2D to show what features each filter detects.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import cv2
import numpy as np

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


# Prints the full model structure showing all layers, their types,
# and their parameter shapes. Useful for identifying layer names
# when accessing specific weights.
#
# Arguments:
# model (MyNetwork) - the trained network to inspect
def print_model_info(model):
    print("Model structure:")
    print(model)
    print()


# Extracts the 10 filters from the first convolutional layer and
# displays them in a 3x4 grid using the viridis colormap. Also prints
# the numeric weight values for each 5x5 filter to the console. Saves
# the resulting plot to a PNG file.
#
# Arguments:
# model (MyNetwork) - the trained network whose conv1 filters to visualize
def visualize_first_layer_filters(model):
    weights = model.conv1.weight.data

    for i in range(weights.shape[0]):
        print(f'Filter {i}:')
        print(weights[i, 0])
        print()

    # Plot the 10 filters in a 3x4 grid
    fig = plt.figure(figsize=(8, 6))
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.imshow(weights[i, 0], cmap='viridis')
        plt.title(f'Filter {i}')
        plt.xticks([])
        plt.yticks([])

    # Turn off axes for empty subplots
    for i in range(10, 12):
        ax = plt.subplot(3, 4, i + 1)
        ax.axis('off')

    plt.suptitle('Conv1 Layer Filters (5x5)')
    plt.tight_layout()
    plt.savefig('conv1_filters.png')
    plt.show()
    print("Conv1 filters saved to conv1_filters.png")


# Applies all 10 conv1 filters to the first training image using
# OpenCV's filter2D function to show the effect of each learned filter.
# Loads the first MNIST training image without normalization so the
# filtered output is visually interpretable. Displays each filter
# alongside its filtered result in a 5x4 grid and saves the plot.
#
# Arguments:
# model (MyNetwork) - the trained network whose conv1 filters to apply
def visualize_filter_effects(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    image, _ = train_dataset[0]
    image_np = image.squeeze(0).numpy()

    # Get the conv1 filter weights without gradients
    with torch.no_grad():
        weights = model.conv1.weight.data

    # Plot filters and their effects in a 5x4 grid
    _, axes = plt.subplots(5, 4, figsize=(8, 10))

    for i in range(10):
        row = i // 2
        col_offset = (i % 2) * 2

        filt = weights[i, 0].numpy()

        filtered = cv2.filter2D(image_np, -1, filt)

        axes[row][col_offset].imshow(filt, cmap='gray')
        axes[row][col_offset].set_title(f'Filter {i}', fontsize=9)
        axes[row][col_offset].set_xticks([])
        axes[row][col_offset].set_yticks([])

        axes[row][col_offset + 1].imshow(filtered, cmap='gray')
        axes[row][col_offset + 1].set_title(f'Filtered {i}', fontsize=9)
        axes[row][col_offset + 1].set_xticks([])
        axes[row][col_offset + 1].set_yticks([])

    plt.suptitle('Conv1 Filters and Their Effects on First Training Image')
    plt.tight_layout()
    plt.savefig('filter_effects.png')
    plt.show()
    print("Filter effects saved to filter_effects.png")


# Main function that orchestrates the network examination. Loads the
# trained model, prints its structure, visualizes the first layer
# filters, and shows their effects on a sample training image.
#
# Arguments:
# argv (list) - command line arguments passed from sys.argv
def main(argv):
    model = load_model('mnist_model.pth')
    print_model_info(model)

    # Visualize the first layer filters
    visualize_first_layer_filters(model)

    # Show the effect of filters on the first training image
    visualize_filter_effects(model)


if __name__ == "__main__":
    main(sys.argv)