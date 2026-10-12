# Furqan Abid
# April 2026
#
# Main script for building, training, and saving a convolutional neural
# network for MNIST digit recognition. Defines the CNN architecture with
# two convolutional layers, dropout, and two fully connected layers.
# Loads the MNIST dataset, plots the first six test digits, trains the
# network for five epochs tracking both training and test metrics, plots
# error curves, and saves the trained model to a file for reuse.

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# CNN network class for MNIST digit recognition. Architecture consists
# of a 5x5 convolution with 10 filters, max pooling with ReLU, a 5x5
# convolution with 20 filters, dropout at 50%, max pooling with ReLU,
# a fully connected layer mapping 320 to 50 with ReLU, and a final
# fully connected layer mapping 50 to 10 with log softmax output.
class MyNetwork(nn.Module):

    # Initializes all network layers with their dimensions and parameters
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # Computes a forward pass through the network by applying
    # convolutions with ReLU and max pooling, flattening, then
    # passing through fully connected layers with log softmax output.
    #
    # Arguments:
    # x (Tensor) - input image tensor of shape (B, 1, 28, 28)
    #
    # Returns:
    # x (Tensor) - log probability tensor of shape (B, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# Trains the network for one complete pass over the training data.
# Computes forward pass, NLL loss, backpropagation, and weight updates
# for each batch. Prints progress at regular intervals and returns the
# average loss and accuracy for the epoch.
#
# Arguments:
# model (MyNetwork)         - the network to train
# train_loader (DataLoader) - DataLoader providing training batches
# optimizer (torch.optim)   - optimizer managing gradient updates
# epoch (int)               - current epoch number for display purposes
# log_interval (int)        - print progress every this many batches (default 100)
#
# Returns:
# avg_loss (float)  - average NLL loss over all training samples
# accuracy (float)  - percentage of correctly classified training samples
def train_network(model, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    total_loss = 0
    correct = 0

    for (batch_idx, (data, target)) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'\nTrain set: Avg. loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)\n')
    return avg_loss, accuracy


# Evaluates the network on the test dataset without computing gradients.
# Returns the average NLL loss and accuracy over all test samples.
#
# Arguments:
# model (MyNetwork)        - the network to evaluate
# test_loader (DataLoader) - DataLoader providing test batches
#
# Returns:
# test_loss (float) - average NLL loss over all test samples
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

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


# Loads the MNIST training and test datasets with standard normalization
# using the precomputed mean and standard deviation. Returns DataLoaders
# for both sets with the specified batch sizes.
#
# Arguments:
# batch_size_train (int) - number of images per training batch (default 64)
# batch_size_test (int)  - number of images per test batch (default 1000)
#
# Returns:
# trainloader (DataLoader) - DataLoader for the shuffled training set
# testloader (DataLoader)  - DataLoader for the unshuffled test set
def load_data(batch_size_train=64, batch_size_test=1000):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    return trainloader, testloader


# Plots training and testing loss and accuracy curves across epochs
# as a two-panel figure. Left panel shows NLL loss for both sets,
# right panel shows accuracy percentages. Saves the plot to a PNG file.
#
# Arguments:
# train_losses (list)     - list of average training losses per epoch
# test_losses (list)      - list of average test losses per epoch
# train_accuracies (list) - list of training accuracies per epoch
# test_accuracies (list)  - list of test accuracies per epoch
def plot_error_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss curves
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Testing Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative Log Likelihood Loss')
    ax1.set_title('Training and Testing Error')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-o', label='Testing Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Testing Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('error_curves.png')
    plt.show()
    print("Error curves saved to error_curves.png")


# Plots the first six example digits from a DataLoader in a 2x3 grid
# with their correct labels displayed above each image. Saves the
# resulting figure to a PNG file.
#
# Arguments:
# data_loader (DataLoader) - DataLoader to pull the first batch from
def plot_first_six(data_loader):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i in range(6):
        row = i // 3
        col = i % 3
        axes[row][col].imshow(example_data[i][0], cmap='gray')
        axes[row][col].set_title(f"Label: {example_targets[i].item()}")
        axes[row][col].axis('off')

    plt.suptitle("First 6 Test Digits from MNIST Dataset")
    plt.tight_layout()
    plt.savefig('first_six_digits.png')
    plt.show()
    print("Plot saved to first_six_digits.png")


# Main function that orchestrates the full MNIST training pipeline.
# Loads data, plots example digits, builds the CNN model, trains for
# five epochs while tracking metrics, plots error curves, and saves
# the trained model and optimizer state to files.
#
# Arguments:
# argv (list) - command line arguments passed from sys.argv
def main(argv):
    n_epochs = 5
    learning_rate = 0.01
    momentum = 0.5

    torch.manual_seed(42)

    print("Loading MNIST data...")
    train_loader, test_loader = load_data()

    print("Plotting the first 6 test digits...")
    plot_first_six(test_loader)

    # Build Network
    model = MyNetwork()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_network(model, train_loader, optimizer, epoch)
        test_loss, test_acc = test_network(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    plot_error_curves(train_losses, test_losses, train_accuracies, test_accuracies)

    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Trained model saved to mnist_model.pth")

    # Save the optimizer state for potential future training continuation
    torch.save(optimizer.state_dict(), 'mnist_optimizer.pth')
    print("Optimizer state saved to mnist_optimizer.pth")

    return


if __name__ == "__main__":
    main(sys.argv)