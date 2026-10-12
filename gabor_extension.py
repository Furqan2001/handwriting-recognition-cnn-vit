# Furqan Abid
# April 2026
#
# Script that replaces the first convolutional layer of the
# MNIST CNN with a bank of hand-crafted Gabor filters at varying
# orientations and frequencies. Freezes the Gabor layer and retrains
# only the remaining layers, then trains an identical network from
# scratch as a baseline. Compares the two approaches by plotting
# filter visualizations, filter effects, and training curves side
# by side to evaluate learned vs hand-crafted feature extraction.

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

# Import the network class from the training script
from mnist_recognition import MyNetwork


# Generates a single 2D Gabor filter kernel by multiplying a Gaussian
# envelope with a sinusoidal carrier at a given orientation and
# frequency. The result is normalized to zero mean and unit variance
# for numerical stability during convolution.
#
# Arguments:
# size (int)        - kernel size in pixels (e.g. 5 for a 5x5 filter)
# theta (float)     - orientation angle of the sinusoid in radians
# frequency (float) - spatial frequency of the sinusoidal component
# sigma (float)     - standard deviation of the Gaussian envelope
# phase (float)     - phase offset of the sinusoidal component
#
# Returns:
# gabor (numpy.ndarray) - 2D array of shape (size, size) containing the filter
def create_gabor_filter(size, theta, frequency, sigma=2.0, phase=0):
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)

    # Rotate coordinates by the orientation angle
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * frequency * x_theta + phase)

    gabor = gaussian * sinusoid

    # Normalize to zero mean and unit variance for stability
    gabor = gabor - gabor.mean()
    if gabor.std() > 0:
        gabor = gabor / gabor.std()

    return gabor


# Creates a bank of 10 Gabor filters combining 5 orientations with
# 2 spatial frequencies. Returns the filters as a tensor shaped to
# match the conv1 weight format [10, 1, 5, 5] so they can be directly
# loaded into the first convolutional layer.
#
# Arguments:
# None
#
# Returns:
# filter_bank (Tensor) - tensor of shape [10, 1, 5, 5] containing the Gabor filters
def create_gabor_filter_bank():
    filters = []

    # 5 orientations x 2 frequencies = 10 filters
    orientations = [0, np.pi/5, 2*np.pi/5, 3*np.pi/5, 4*np.pi/5]
    frequencies = [0.3, 0.5]

    for freq in frequencies:
        for theta in orientations:
            gabor = create_gabor_filter(size=5, theta=theta, frequency=freq)
            filters.append(gabor)

    filter_bank = np.array(filters).reshape(10, 1, 5, 5)
    return torch.tensor(filter_bank, dtype=torch.float32)


# Displays the 10 Gabor filters on the top row and the 10 learned
# conv1 filters on the bottom row for side-by-side comparison. Saves
# the resulting figure to a PNG file.
#
# Arguments:
# gabor_filters (Tensor)   - Gabor filter weights of shape [10, 1, 5, 5]
# learned_filters (Tensor) - learned conv1 weights of shape [10, 1, 5, 5]
def plot_filter_comparison(gabor_filters, learned_filters):
    _, axes = plt.subplots(2, 10, figsize=(16, 4))

    for i in range(10):
        # Top row: Gabor filters
        axes[0][i].imshow(gabor_filters[i, 0].numpy(), cmap='gray')
        axes[0][i].set_title(f'{i}', fontsize=9)
        axes[0][i].set_xticks([])
        axes[0][i].set_yticks([])

        # Bottom row: Learned filters
        axes[1][i].imshow(learned_filters[i, 0].numpy(), cmap='gray')
        axes[1][i].set_title(f'{i}', fontsize=9)
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])

    axes[0][0].set_ylabel('Gabor', fontsize=11)
    axes[1][0].set_ylabel('Learned', fontsize=11)

    plt.suptitle('Gabor Filters vs Learned Conv1 Filters', fontsize=14)
    plt.tight_layout()
    plt.savefig('gabor_vs_learned_filters.png')
    plt.show()
    print("Filter comparison saved to gabor_vs_learned_filters.png")


# Applies each of the 10 Gabor filters to the first MNIST training
# image using OpenCV's filter2D and displays the filters alongside
# their filtered outputs in a 5x4 grid. Saves the plot to a file.
#
# Arguments:
# gabor_filters (Tensor) - Gabor filter weights of shape [10, 1, 5, 5]
def plot_gabor_filter_effects(gabor_filters):
    # Load the first training image
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    image, label = train_dataset[0]
    image_np = image.squeeze(0).numpy()

    _, axes = plt.subplots(5, 4, figsize=(8, 10))

    for i in range(10):
        row = i // 2
        col_offset = (i % 2) * 2

        filt = gabor_filters[i, 0].numpy()
        filtered = cv2.filter2D(image_np, -1, filt)

        axes[row][col_offset].imshow(filt, cmap='gray')
        axes[row][col_offset].set_title(f'Gabor {i}', fontsize=9)
        axes[row][col_offset].set_xticks([])
        axes[row][col_offset].set_yticks([])

        axes[row][col_offset + 1].imshow(filtered, cmap='gray')
        axes[row][col_offset + 1].set_title(f'Filtered {i}', fontsize=9)
        axes[row][col_offset + 1].set_xticks([])
        axes[row][col_offset + 1].set_yticks([])

    plt.suptitle('Gabor Filter Effects on First Training Image')
    plt.tight_layout()
    plt.savefig('gabor_filter_effects.png')
    plt.show()
    print("Gabor filter effects saved to gabor_filter_effects.png")


# Loads the MNIST training and test datasets with standard normalization
# and returns DataLoaders for both sets.
#
# Arguments:
# batch_size_train (int) - number of images per training batch (default 64)
# batch_size_test (int)  - number of images per test batch (default 1000)
#
# Returns:
# train_loader (DataLoader) - DataLoader for the training set
# test_loader (DataLoader)  - DataLoader for the test set
def load_data(batch_size_train=64, batch_size_test=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader


# Trains the network for one complete pass over the training data.
# Computes forward pass, backpropagation, and weight updates for each
# batch. Returns the average loss and accuracy across the full epoch.
#
# Arguments:
# model (MyNetwork)        - the network to train
# train_loader (DataLoader) - DataLoader providing training batches
# optimizer (torch.optim)  - optimizer managing gradient updates
#
# Returns:
# avg_loss (float)  - average NLL loss over all training samples
# accuracy (float)  - percentage of correctly classified training samples
def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for (data, target) in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy


# Evaluates the network on the test dataset without computing gradients.
# Returns the average loss and accuracy over all test samples.
#
# Arguments:
# model (MyNetwork)       - the network to evaluate
# test_loader (DataLoader) - DataLoader providing test batches
#
# Returns:
# avg_loss (float)  - average NLL loss over all test samples
# accuracy (float)  - percentage of correctly classified test samples
def test_network(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy


# Plots training and test loss and accuracy curves for both the Gabor
# network and the fully learned network on the same axes for direct
# comparison. Saves the resulting two-panel figure to a PNG file.
#
# Arguments:
# gabor_losses (list)        - Gabor network training losses per epoch
# gabor_accs (list)          - Gabor network training accuracies per epoch
# learned_losses (list)      - learned network training losses per epoch
# learned_accs (list)        - learned network training accuracies per epoch
# gabor_test_losses (list)   - Gabor network test losses per epoch
# gabor_test_accs (list)     - Gabor network test accuracies per epoch
# learned_test_losses (list) - learned network test losses per epoch
# learned_test_accs (list)   - learned network test accuracies per epoch
def plot_training_comparison(gabor_losses, gabor_accs, learned_losses, learned_accs,
                              gabor_test_losses, gabor_test_accs,
                              learned_test_losses, learned_test_accs):
    epochs = range(1, len(gabor_losses) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss comparison
    ax1.plot(epochs, gabor_losses, 'b-o', markersize=4, label='Gabor - Train')
    ax1.plot(epochs, gabor_test_losses, 'b--s', markersize=4, label='Gabor - Test')
    ax1.plot(epochs, learned_losses, 'r-o', markersize=4, label='Learned - Train')
    ax1.plot(epochs, learned_test_losses, 'r--s', markersize=4, label='Learned - Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training/Test Loss: Gabor vs Learned Filters')
    ax1.legend()
    ax1.grid(True)

    # Accuracy comparison
    ax2.plot(epochs, gabor_accs, 'b-o', markersize=4, label='Gabor - Train')
    ax2.plot(epochs, gabor_test_accs, 'b--s', markersize=4, label='Gabor - Test')
    ax2.plot(epochs, learned_accs, 'r-o', markersize=4, label='Learned - Train')
    ax2.plot(epochs, learned_test_accs, 'r--s', markersize=4, label='Learned - Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training/Test Accuracy: Gabor vs Learned Filters')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('gabor_vs_learned_training.png')
    plt.show()
    print("Training comparison saved to gabor_vs_learned_training.png")


# Main function that orchestrates the Gabor filter experiment.
# Creates a Gabor filter bank, builds two networks (one with frozen
# Gabor conv1 and one fully learned from scratch), trains both for 5
# epochs, prints a comparison of their final test accuracies, and
# generates filter visualization and training curve plots.
#
# Arguments:
# argv (list) - command line arguments passed from sys.argv
def main(argv):
    n_epochs = 5
    learning_rate = 0.01
    momentum = 0.5

    torch.manual_seed(42)

    # Load data
    train_loader, test_loader = load_data()

    # Create the Gabor filter bank
    gabor_filters = create_gabor_filter_bank()

    # Network 1: Gabor filters (fixed conv1, train the rest)
    print("TRAINING NETWORK 1: Gabor filter first layer (fixed)")

    gabor_model = MyNetwork()

    # Replace conv1 weights with Gabor filters and freeze them
    with torch.no_grad():
        gabor_model.conv1.weight.copy_(gabor_filters)
        gabor_model.conv1.bias.zero_()

    gabor_model.conv1.weight.requires_grad = False
    gabor_model.conv1.bias.requires_grad = False

    gabor_optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, gabor_model.parameters()),
        lr=learning_rate, momentum=momentum
    )

    # Train
    gabor_train_losses, gabor_train_accs = [], []
    gabor_test_losses, gabor_test_accs = [], []

    for _ in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(gabor_model, train_loader, gabor_optimizer)
        test_loss, test_acc = test_network(gabor_model, test_loader)
        gabor_train_losses.append(train_loss)
        gabor_train_accs.append(train_acc)
        gabor_test_losses.append(test_loss)
        gabor_test_accs.append(test_acc)

    # Network 2: Fully learned (baseline, train from scratch)
    print("TRAINING NETWORK 2: Fully learned (baseline)")
    print()

    torch.manual_seed(42)

    learned_model = MyNetwork()
    learned_optimizer = optim.SGD(learned_model.parameters(),
                                   lr=learning_rate, momentum=momentum)

    learned_train_losses, learned_train_accs = [], []
    learned_test_losses, learned_test_accs = [], []

    for _ in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(learned_model, train_loader, learned_optimizer)
        test_loss, test_acc = test_network(learned_model, test_loader)
        learned_train_losses.append(train_loss)
        learned_train_accs.append(train_acc)
        learned_test_losses.append(test_loss)
        learned_test_accs.append(test_acc)

    # Comparison and visualization
    print("=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)

    print(f"Gabor filter network:  {gabor_test_accs[-1]:.2f}% test accuracy")
    print(f"Fully learned network: {learned_test_accs[-1]:.2f}% test accuracy")

    # Plot filter comparison and effects
    plot_filter_comparison(gabor_filters, learned_model.conv1.weight.data)
    plot_gabor_filter_effects(gabor_filters)

    # Plot training comparison
    plot_training_comparison(
        gabor_train_losses, gabor_train_accs,
        learned_train_losses, learned_train_accs,
        gabor_test_losses, gabor_test_accs,
        learned_test_losses, learned_test_accs
    )

    torch.save(gabor_model.state_dict(), 'gabor_model.pth')
    print()
    print("Gabor model saved to gabor_model.pth")


if __name__ == "__main__":
    main(sys.argv)