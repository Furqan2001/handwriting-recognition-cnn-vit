# Furqan Abid
# April 2026
#
# Script for building, training, and saving a Vision Transformer (ViT)
# for MNIST digit recognition. Defines the transformer architecture
# including patch embedding, positional encoding, stacked transformer
# encoder layers, token pooling, and a classification MLP. Loads the
# MNIST dataset, trains the model for a configurable number of epochs,
# plots training and test error curves, and saves the trained model.

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Configuration class that holds all hyperparameters and architecture
# settings for the Vision Transformer. Stores fixed dataset attributes
# like image size and number of classes, as well as variable settings
# like patch size, embedding dimension, depth, and training parameters.
# Also builds a CSV-formatted config string for logging results.
#
# Arguments:
# name (str)           - identifier for this configuration
# dataset (str)        - name of the dataset being used
# patch_size (int)     - size of each image patch in pixels
# stride (int)         - stride between patches (overlap if < patch_size)
# embed_dim (int)      - dimensionality of token embeddings
# depth (int)          - number of stacked transformer encoder layers
# num_heads (int)      - number of attention heads per layer
# mlp_dim (int)        - hidden dimension of the feedforward network
# dropout (float)      - dropout rate applied throughout the network
# use_cls_token (bool) - whether to use a CLS token or mean pooling
# epochs (int)         - number of training epochs
# batch_size (int)     - number of images per training batch
# lr (float)           - learning rate for the optimizer
# weight_decay (float) - L2 regularization weight decay
# seed (int)           - random seed for reproducibility
# optimizer (str)      - optimizer type ('adamw' or 'sgd')
# device (str)         - device to train on ('cpu', 'cuda', or 'mps')
class NetConfig:

    def __init__(self,
                 name='vit_base',
                 dataset='mnist',
                 patch_size=4,
                 stride=2,
                 embed_dim=48,
                 depth=4,
                 num_heads=8,
                 mlp_dim=128,
                 dropout=0.1,
                 use_cls_token=False,
                 epochs=15,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-4,
                 seed=0,
                 optimizer='adamw',
                 device='cpu',
                 ):

        # data set fixed attributes
        self.image_size = 28
        self.in_channels = 1
        self.num_classes = 10

        # variable things
        self.name = name
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_cls_token = use_cls_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.optimizer = optimizer
        self.device = device

        s = "Name,Dataset,PatchSize,Stride,Dim,Depth,Heads,MLPDim,Dropout,CLS,Epochs,Batch,LR,Decay,Seed,Optimizer,TestAcc,BestEpoch\n"
        s += "%s,%s,%d,%d,%d,%d,%d,%d,%0.2f,%s,%d,%d,%f,%f,%d,%s," % (
            self.name,
            self.dataset,
            self.patch_size,
            self.stride,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.mlp_dim,
            self.dropout,
            self.use_cls_token,
            self.epochs,
            self.batch_size,
            self.lr,
            self.weight_decay,
            self.seed,
            self.optimizer
        )
        self.config_string = s

        return


# Patch embedding module that converts an image into a sequence of
# token embeddings. Uses nn.Unfold to extract sliding local patches
# from the image, flattens each patch into a vector, and projects it
# into the embedding space via a linear layer. Supports both overlapping
# and non-overlapping patches depending on stride relative to patch size.
#
# Arguments:
# image_size (int)   - spatial dimension of the square input image
# patch_size (int)   - size of each patch in pixels
# stride (int)       - stride between adjacent patches
# in_channels (int)  - number of input image channels
# embed_dim (int)    - dimensionality of the output token embeddings
class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, stride, in_channels, embed_dim):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Unfold extracts sliding local blocks from the image
        self.unfold = nn.Unfold(
            kernel_size=patch_size,
            stride=stride,
        )

        # Each extracted patch is flattened into one vector of this size
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection from flattened patch to embedding dimension
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

        # Precompute how many patches will be produced
        self.num_patches = self._compute_num_patches()

    # Computes total number of patches produced by the unfold operation
    # based on image size, patch size, and stride.
    #
    # Returns:
    # (int) - total number of patches (positions_per_dim squared)
    def _compute_num_patches(self):
        positions_per_dim = ((self.image_size - self.patch_size) // self.stride) + 1
        return positions_per_dim * positions_per_dim

    # Extracts patches from the input image, transposes so each patch
    # is a row, and projects into embedding space.
    #
    # Arguments:
    # x (Tensor) - input image tensor of shape (B, C, H, W)
    #
    # Returns:
    # x (Tensor) - patch embeddings of shape (B, N, embed_dim)
    def forward(self, x):
        x = self.unfold(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


# Vision Transformer network for MNIST digit recognition. Architecture
# consists of patch embedding, learnable positional encoding, a stack
# of transformer encoder layers with multi-head self-attention and
# feedforward networks, token pooling (either CLS token or mean
# pooling), layer normalization, and a classification MLP that outputs
# log probabilities over the digit classes.
class NetTransformer(nn.Module):

    # Initializes all transformer network layers based on the provided
    # configuration including patch embedding, positional encoding,
    # transformer encoder stack, normalization, and classifier MLP.
    #
    # Arguments:
    # config (NetConfig) - configuration object with all hyperparameters
    def __init__(self, config):
        super(NetTransformer, self).__init__()

        # Create the patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # How many tokens (patches) are there?
        num_tokens = self.patch_embed.num_patches
        print("Number of tokens: %d" % (num_tokens))

        # Does it use a CLS token or global average pooling?
        self.use_cls_token = config.use_cls_token

        # If using a CLS token, create a learnable parameter for it
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = num_tokens + 1
        else:
            self.cls_token = None
            total_tokens = num_tokens

        # Learnable positional embedding, one for each token position
        self.pos_embed = nn.Parameter(
            torch.zeros(1, total_tokens, config.embed_dim))
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer encoder layer with multi-head self attention,
        # feedforward network, layer norm, residual connections, dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        # Stack of transformer layers forming the encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.depth,
        )

        # Final normalization layer before classification
        self.norm = nn.LayerNorm(config.embed_dim)

        # Classification MLP: linear -> GELU -> linear
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.num_classes)
        )

        return

    # Initializes positional embedding and optional CLS token parameters
    # using truncated normal distribution for stable training.
    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    # Computes a forward pass through the transformer network. Converts
    # the input image to patch embeddings, adds positional encoding,
    # runs through the transformer encoder, pools tokens into a single
    # vector, normalizes, and classifies.
    #
    # Arguments:
    # x (Tensor) - input image tensor of shape (B, 1, 28, 28)
    #
    # Returns:
    # (Tensor) - log probability tensor of shape (B, num_classes)
    def forward(self, x):
        # Execute the patch embedding layer
        x = self.patch_embed(x)

        # Get the batch size
        batch_size = x.size(0)

        # Add the optional CLS token to the beginning of the sequence
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # Add the learnable positional embedding to each token
        x = x + self.pos_embed

        # Run the dropout layer after patch embedding + positional encoding
        x = self.pos_dropout(x)

        # Run the transformer encoder
        x = self.encoder(x)

        # Either use the CLS token or average pool all tokens
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        # Final normalization of the token to classify
        x = self.norm(x)

        # Call the classification MLP
        x = self.classifier(x)

        # Return the log softmax of the output layer
        return F.log_softmax(x, dim=1)


# Loads the MNIST training and test datasets with standard normalization
# and returns DataLoaders for both sets with the specified batch sizes.
#
# Arguments:
# batch_size_train (int) - number of images per training batch (default 64)
# batch_size_test (int)  - number of images per test batch (default 1000)
#
# Returns:
# train_loader (DataLoader) - DataLoader for the shuffled training set
# test_loader (DataLoader)  - DataLoader for the unshuffled test set
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


# Trains the transformer network for one complete pass over the
# training data. Moves each batch to the specified device, computes
# forward pass, NLL loss, backpropagation, and weight updates. Prints
# progress at regular intervals and returns average loss and accuracy.
#
# Arguments:
# model (NetTransformer)    - the transformer network to train
# train_loader (DataLoader) - DataLoader providing training batches
# optimizer (torch.optim)   - optimizer managing gradient updates
# epoch (int)               - current epoch number for display purposes
# device (torch.device)     - device to run computations on
# log_interval (int)        - print progress every this many batches (default 100)
#
# Returns:
# avg_loss (float)  - average NLL loss over all training samples
# accuracy (float)  - percentage of correctly classified training samples
def train_network(model, train_loader, optimizer, epoch, device, log_interval=100):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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


# Evaluates the transformer network on the test dataset without
# computing gradients. Returns the average NLL loss and accuracy
# over all test samples.
#
# Arguments:
# model (NetTransformer)   - the transformer network to evaluate
# test_loader (DataLoader) - DataLoader providing test batches
# device (torch.device)    - device to run computations on
#
# Returns:
# test_loss (float) - average NLL loss over all test samples
# accuracy (float)  - percentage of correctly classified test samples
def test_network(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Avg. loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


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

    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Testing Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative Log Likelihood Loss')
    ax1.set_title('Transformer - Training and Testing Error')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-o', label='Testing Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Transformer - Training and Testing Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/transformer_error_curves.png')
    plt.show()
    print("Error curves saved to outputs/transformer_error_curves.png")


# Main function that orchestrates the full Vision Transformer training
# pipeline. Creates a configuration, selects the compute device, loads
# MNIST data, builds the transformer model, trains for the configured
# number of epochs while tracking metrics, plots error curves, saves
# the model, and prints the configuration summary.
#
# Arguments:
# argv (list) - command line arguments passed from sys.argv
def main(argv):
    os.makedirs('outputs', exist_ok=True)

    # Create the configuration with default settings
    config = NetConfig()

    # Set device (use cpu if mps/cuda not available)
    if config.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # Load the MNIST data
    print("Loading MNIST data...")
    train_loader, test_loader = load_data(config.batch_size)

    # Build the transformer model
    print("\nBuilding Vision Transformer model...")
    model = NetTransformer(config)
    model.to(device)
    print(model)
    print()

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Set up the optimizer
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                                weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    # Track losses and accuracies for plotting
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Train the network
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_network(model, train_loader, optimizer,
                                               epoch, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = test_network(model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Plot error curves
    plot_error_curves(train_losses, test_losses, train_accuracies, test_accuracies)

    # Save the trained model
    torch.save(model.state_dict(), 'models/transformer_model.pth')
    print("Model saved to models/transformer_model.pth")

    # Print config summary
    print("\nConfiguration:")
    print(config.config_string)


if __name__ == "__main__":
    main(sys.argv)