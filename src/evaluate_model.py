# Furqan Abid
# April 2026
#
# Script that loads a trained MNIST CNN model from a file and evaluates
# it on the first 10 examples of the test set. For each example, prints
# the 10 network output values, the predicted digit, and the correct
# label. Also generates a 3x3 grid plot of the first 9 test digits
# with their predicted labels displayed above each image.

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import the network class from the training script
from mnist_recognition import MyNetwork


# Loads the MNIST test dataset with no shuffling so the ordering is
# consistent across runs. Returns a DataLoader with batch_size=1 for
# individual example access.
#
# Arguments:
# None
#
# Returns:
# test_loader (DataLoader) - DataLoader yielding one test image at a time
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    # Use batch_size=1 so we can iterate one example at a time
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader


# Loads the trained MNIST model from a saved state dictionary file and
# sets it to evaluation mode so that dropout is disabled and outputs
# are deterministic.
#
# Arguments:
# model_path (str) - file path to the saved model state dictionary
#
# Returns:
# model (MyNetwork) - the trained network ready for inference
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


# Runs the model on the first n examples from the test set. For each
# example, prints the 10 output log-probability values formatted to
# two decimal places, the index of the maximum output value as the
# prediction, and the correct label. Returns collected data for plotting.
#
# Arguments:
# model (MyNetwork)      - the trained network in evaluation mode
# test_loader (DataLoader) - DataLoader yielding one test image at a time
# n (int)                - number of examples to evaluate (default 10)
#
# Returns:
# images (list)      - list of 28x28 tensors for each evaluated image
# predictions (list) - list of predicted digit labels
# labels (list)      - list of correct digit labels
def evaluate_first_n(model, test_loader, n=10):
    images = []
    predictions = []
    labels = []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= n:
                break

            output = model(data)

            # Get the predicted digit (index of max output value)
            pred = output.argmax(dim=1, keepdim=True).item()
            label = target.item()

            # Store for plotting
            images.append(data.squeeze(0).squeeze(0))
            predictions.append(pred)
            labels.append(label)

            # Print the 10 output values with 2 decimal places
            output_values = output.squeeze().tolist()
            output_str = ' '.join([f'{v:7.2f}' for v in output_values])
            print(f'Example {i}: Output: [{output_str}]')
            print(f'           Predicted: {pred}, Correct: {label}, '
                  f'{"CORRECT" if pred == label else "WRONG"}')
            print()

    return images, predictions, labels


# Plots the first 9 test digits in a 3x3 grid with the network
# prediction shown as the title above each image. Correct predictions
# are displayed in green and incorrect ones in red. Saves the resulting
# figure to a PNG file.
#
# Arguments:
# images (list)      - list of 28x28 tensors for each image
# predictions (list) - list of predicted digit labels
# labels (list)      - list of correct digit labels
def plot_predictions(images, predictions, labels):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        row = i // 3
        col = i % 3
        axes[row][col].imshow(images[i], cmap='gray')
        # Show prediction in the title, mark incorrect predictions
        correct = predictions[i] == labels[i]
        color = 'green' if correct else 'red'
        axes[row][col].set_title(f'Prediction: {predictions[i]}', color=color, fontsize=12)
        axes[row][col].axis('off')

    plt.suptitle('First 9 Test Digits with Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/test_predictions.png')
    plt.show()
    print("Predictions plot saved to outputs/test_predictions.png")


# Main function that orchestrates the evaluation pipeline. Loads the
# trained model from file, loads the test dataset, evaluates the first
# 10 examples printing detailed output values, and generates a plot
# of the first 9 predictions.
#
# Arguments:
# argv (list) - command line arguments passed from sys.argv
def main(argv):
    os.makedirs('outputs', exist_ok=True)

    model = load_model('models/mnist_model.pth')
    test_loader = load_test_data()

    # Evaluate the first 10 examples
    print("Evaluating first 10 test examples:\n")
    images, predictions, labels = evaluate_first_n(model, test_loader, n=10)

    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    print(f"Results: {correct}/10 correct\n")
    plot_predictions(images, predictions, labels)


if __name__ == "__main__":
    main(sys.argv)