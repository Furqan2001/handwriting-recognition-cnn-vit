# Furqan Abid
# April 2026
#
# Transfer learning script that re-uses the pretrained MNIST digit
# recognition CNN to classify three Greek letters: alpha, beta, and
# gamma. Loads the MNIST model, freezes all existing weights, replaces
# the final layer with a 3-class output, and trains only that layer
# on 27 example images. Also supports testing on custom handwritten
# Greek letter images provided in a separate directory.

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

from mnist_recognition import MyNetwork


# Custom transform class that converts Greek letter images to match
# the MNIST format. Applies grayscale conversion, affine scaling to
# shrink the image, center cropping to 28x28, and intensity inversion
# so that digits appear white on a black background.
#
# Arguments:
# x (Tensor) - input image tensor to transform
#
# Returns:
# x (Tensor) - transformed image tensor matching MNIST format
class GreekTransform:
    def __init__(self):
        pass

    # Applies the full transformation pipeline to a single image
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Loads the pretrained MNIST model from a saved state dictionary,
# freezes all network parameters so they will not be updated during
# training, and replaces the final fully connected layer with a new
# Linear layer mapping 50 inputs to 3 outputs for the three Greek
# letter classes. Prints the modified model structure.
#
# Arguments:
# model_path (str) - file path to the saved MNIST model state dictionary
#
# Returns:
# model (MyNetwork) - modified network with frozen weights and new output layer
def build_greek_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, weights_only=True))

    for param in model.parameters():
        param.requires_grad = False

    model.fc2 = nn.Linear(50, 3)

    print("Modified model for Greek letter recognition:")
    print(model)
    print()

    return model


# Creates a DataLoader for the Greek letter training set using the
# ImageFolder class, which automatically assigns labels based on
# subfolder names (alpha, beta, gamma). Applies ToTensor, the custom
# GreekTransform, and MNIST normalization to each image.
#
# Arguments:
# training_set_path (str) - path to directory containing alpha/, beta/, gamma/ folders
#
# Returns:
# greek_train (DataLoader) - DataLoader yielding batches of 5 transformed images
def load_greek_data(training_set_path):
    greek_train = DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )

    return greek_train


# Trains the model for one epoch over the Greek letter training data.
# Computes forward pass, NLL loss, backpropagation, and weight updates
# for each batch. Prints and returns the average loss and accuracy
# for the epoch.
#
# Arguments:
# model (MyNetwork)         - the network to train
# train_loader (DataLoader) - DataLoader providing Greek letter batches
# optimizer (torch.optim)   - optimizer managing gradient updates for fc2
# epoch (int)               - current epoch number for display purposes
#
# Returns:
# avg_loss (float)  - average NLL loss over all training samples
# accuracy (float)  - percentage of correctly classified training samples
def train_greek_model(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.1f}%)')
    return avg_loss, accuracy


# Plots the training loss and accuracy curves over epochs as a
# two-panel figure. Left panel shows NLL loss decreasing, right panel
# shows accuracy increasing. Saves the plot to a PNG file.
#
# Arguments:
# losses (list)     - list of average training losses per epoch
# accuracies (list) - list of training accuracies per epoch
def plot_greek_training(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, losses, 'b-o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Greek Letters - Training Loss')
    ax1.grid(True)

    ax2.plot(epochs, accuracies, 'g-o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Greek Letters - Training Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('greek_training_curves.png')
    plt.show()
    print("Training curves saved to greek_training_curves.png")


# Preprocesses a custom Greek letter image for classification. Resizes
# to 128x128 to match the expected input size of GreekTransform, then
# applies the same transform pipeline used during training. Returns
# both a normalized tensor for the network and a display-friendly
# version without normalization for plotting.
#
# Arguments:
# image_path (str) - file path to the custom Greek letter image
#
# Returns:
# tensor (Tensor)      - normalized image tensor with batch dimension for inference
# image (Tensor)       - unnormalized 28x28 tensor for display purposes
def preprocess_custom_image(image_path):
    img = Image.open(image_path).convert('RGB')

    img = img.resize((128, 128), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img).unsqueeze(0)

    # Also get a displayable version (without normalization)
    display_transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform()
    ])
    image = display_transform(img).squeeze(0)

    return tensor, image


# Tests the trained model on custom Greek letter images from a given
# directory. For each image, preprocesses it and runs a forward pass
# to get a prediction. Displays all images in a grid with predicted
# class names as titles and saves the plot to a PNG file.
#
# Arguments:
# model (MyNetwork) - the trained Greek letter network in evaluation mode
# image_dir (str)   - path to directory containing custom Greek letter images
def test_custom_greek_images(model, image_dir):
    # Class names corresponding to folder order (alphabetical)
    class_names = ['alpha', 'beta', 'gamma']

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_extensions)
    ])

    if not image_files:
        print(f"No images found in '{image_dir}'.")
        return

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    images = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for path in image_paths:
            tensor, display_image = preprocess_custom_image(path)
            output = model(tensor)
            pred = output.argmax(dim=1, keepdim=True).item()

            images.append(display_image)
            predictions.append(pred)

    # Plot results
    n = len(images)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        if i < n:
            axes[row][col].imshow(images[i], cmap='gray')
            axes[row][col].set_title(f'Pred: {class_names[predictions[i]]}', fontsize=11)
        axes[row][col].axis('off')

    plt.suptitle('Custom Greek Letters - Network Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig('custom_greek_predictions.png')
    plt.show()
    print("Custom Greek predictions saved to custom_greek_predictions.png")


# Main function that orchestrates the Greek letter transfer learning
# pipeline. Builds the modified model with frozen MNIST weights, loads
# the training data, trains only the new output layer for up to 50
# epochs or until perfect accuracy, plots training curves, saves the
# model, and optionally tests on custom handwritten Greek letter images.
#
# Arguments:
# argv (list) - command line arguments: [training_path] [custom_test_path]
def main(argv):
    training_set_path = 'greek_train'
    custom_test_path = 'greek_custom'
    n_epochs = 50
    learning_rate = 0.01
    momentum = 0.5

    if len(argv) > 1:
        training_set_path = argv[1]
    if len(argv) > 2:
        custom_test_path = argv[2]

    torch.manual_seed(42)

    model = build_greek_model()
    greek_train = load_greek_data(training_set_path)

    optimizer = optim.SGD(model.fc2.parameters(), lr=learning_rate, momentum=momentum)

    losses = []
    accuracies = []

    for epoch in range(1, n_epochs + 1):
        loss, acc = train_greek_model(model, greek_train, optimizer, epoch)
        losses.append(loss)
        accuracies.append(acc)

        if acc >= 100.0:
            print(f"Reached 100% accuracy at epoch {epoch}. Stopping training.")
            break

    plot_greek_training(losses, accuracies)

    torch.save(model.state_dict(), 'greek_model.pth')
    print("Greek model saved to greek_model.pth\n")

    # Test on custom Greek letter images if directory exists
    if os.path.isdir(custom_test_path):
        print(f"\nTesting on custom Greek letters from '{custom_test_path}'...\n")
        test_custom_greek_images(model, custom_test_path)
    else:
        print(f"\nNo custom test directory found at '{custom_test_path}'.")
        print("To test custom images, run: python greek_recognition.py greek_train greek_custom")


if __name__ == "__main__":
    main(sys.argv)