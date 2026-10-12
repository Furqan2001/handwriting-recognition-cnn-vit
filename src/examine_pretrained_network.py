# Furqan Abid
# April 2026
#
# Script that loads a pretrained VGG16 network from PyTorch's
# model zoo and explores its first two convolutional layers. Visualizes
# the learned 3x3 filters, applies them to an MNIST digit image to show
# their effects, and passes the image through the network to display
# intermediate activation maps from the first two conv+ReLU stages.

import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from PIL import Image


# Loads the pretrained VGG16 model using ImageNet weights and sets it
# to evaluation mode. Prints the feature extraction layer structure
# so the user can identify layer indices for weight access.
#
# Arguments:
# None
#
# Returns:
# model (torchvision.models.VGG) - pretrained VGG16 in evaluation mode
def load_pretrained_vgg16():
    print("Loading pretrained VGG16...")
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    return model


# Visualizes up to 16 filters from a given convolutional layer in a
# grid. For layers with multiple input channels, the filter weights
# are averaged across input channels to produce a single 2D image
# per filter. Saves the resulting plot to a PNG file.
#
# Arguments:
# weights (Tensor)    - filter weight tensor of shape [N, C, H, W]
# layer_name (str)    - descriptive name for the layer used in the plot title
# filename (str)      - output file path for the saved plot
def visualize_conv_filters(weights, layer_name, filename):
    num_filters = weights.shape[0]
    # Show up to 16 filters in a 4x4 grid (or smaller if fewer filters)
    n = min(num_filters, 16)
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        if i < n:
            # For multi-channel filters, average across input channels for visualization
            if weights.shape[1] > 1:
                filt = weights[i].mean(dim=0).numpy()
            else:
                filt = weights[i, 0].numpy()
            axes[row][col].imshow(filt, cmap='gray')
            axes[row][col].set_title(f'{i}', fontsize=8)
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

    plt.suptitle(f'{layer_name} Filters ({weights.shape[0]} total, showing {n})', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved to {filename}")


# Applies the first convolutional layer's filters to a grayscale image
# using OpenCV's filter2D and displays the filtered outputs in a grid.
# For multi-channel filters, the weights are averaged across input
# channels to produce a single 2D kernel for convolution. Shows up to
# 16 filtered results and saves the plot.
#
# Arguments:
# model (torchvision.models.VGG) - pretrained VGG16 model
# image_np (numpy.ndarray)       - 2D grayscale image array to filter
# layer_name (str)               - descriptive name for the plot title
# filename (str)                 - output file path for the saved plot
def visualize_filter_effects(model, image_np, layer_name, filename):
    # Get the first conv layer
    conv_layer = None
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            conv_layer = layer
            break

    weights = conv_layer.weight.data

    n = min(weights.shape[0], 16)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))

    with torch.no_grad():
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            if i < n:
                # Average the filter across input channels to get a single 3x3 kernel
                filt = weights[i].mean(dim=0).numpy()
                filtered = cv2.filter2D(image_np, -1, filt)
                axes[row][col].imshow(filtered, cmap='gray')
                axes[row][col].set_title(f'Filter {i}', fontsize=9)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    plt.suptitle(f'{layer_name} Filter Effects on MNIST Digit', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved to {filename}")


# Passes an image tensor through the first several layers of VGG16 and
# captures the intermediate activation maps after the first and second
# conv+ReLU stages. Displays the first 8 activation channels for each
# layer in a 2x8 grid, showing what features the network detects at
# different depths. Saves the plot to a file.
#
# Arguments:
# model (torchvision.models.VGG) - pretrained VGG16 model
# image_tensor (Tensor)          - preprocessed input tensor of shape (1, 3, 224, 224)
# filename (str)                 - output file path for the saved plot
def visualize_activations(model, image_tensor, filename):
    # Get activations after first conv+relu and second conv+relu
    activations = []
    layer_names = []
    x = image_tensor

    with torch.no_grad():
        for i, layer in enumerate(model.features):
            x = layer(x)
            # Capture output after ReLU layers (indices 1 and 3 in VGG16)
            if i == 1:  # After first conv + relu
                activations.append(x.squeeze(0))
                layer_names.append('After Conv1 + ReLU')
            elif i == 3:  # After second conv + relu
                activations.append(x.squeeze(0))
                layer_names.append('After Conv2 + ReLU')
            if i > 4:
                break

    # Plot first 16 activation maps for each layer
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))

    for layer_idx in range(2):
        act = activations[layer_idx]
        for i in range(8):
            axes[layer_idx][i].imshow(act[i].numpy(), cmap='gray')
            axes[layer_idx][i].set_xticks([])
            axes[layer_idx][i].set_yticks([])
            if i == 0:
                axes[layer_idx][i].set_ylabel(layer_names[layer_idx], fontsize=9)

    plt.suptitle('VGG16 Activations on MNIST Digit (First 8 Channels)', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved to {filename}")


# Main function that orchestrates the VGG16 exploration. Loads the
# pretrained model, visualizes filters from the first two convolutional
# layers, applies them to an MNIST digit, and displays intermediate
# activation maps by passing the digit through the network.
#
# Arguments:
# argv (list) - command line arguments passed from sys.argv
def main(argv):
    os.makedirs('outputs', exist_ok=True)

    # Load pretrained VGG16
    model = load_pretrained_vgg16()

    # Get conv layer weights
    conv1_weights = model.features[0].weight.data
    conv2_weights = model.features[2].weight.data

    # Visualize conv1 and conv2 filters (first 16)
    print("Visualizing Conv1 filters...")
    visualize_conv_filters(conv1_weights, 'VGG16 Conv1', 'outputs/vgg16_conv1_filters.png')
    print("Visualizing Conv2 filters...")
    visualize_conv_filters(conv2_weights, 'VGG16 Conv2', 'outputs/vgg16_conv2_filters.png')

    # Load an MNIST digit for visualization
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    image, label = train_dataset[0]
    image_np = image.squeeze(0).numpy()
    print(f"\nUsing MNIST digit '{label}' for filter effect visualization")

    visualize_filter_effects(model, image_np, 'VGG16 Conv1', 'outputs/vgg16_filter_effects.png')

    # For activations, VGG16 expects 3-channel 224x224 input
    # Resize and replicate the MNIST digit to 3 channels
    img_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    img_pil = img_pil.resize((224, 224), Image.LANCZOS)

    vgg_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    vgg_input = vgg_transform(img_pil).unsqueeze(0)

    print("Visualizing layer activations...")
    visualize_activations(model, vgg_input, 'outputs/vgg16_activations.png')


if __name__ == "__main__":
    main(sys.argv)