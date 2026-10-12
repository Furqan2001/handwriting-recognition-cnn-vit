# Furqan Abid
# April 2026
#
# Experiment script that automates evaluation of Vision Transformer
# architecture variations on the Fashion MNIST dataset. Tests across
# three dimensions: transformer depth, embedding dimension, and number
# of attention heads. Uses a round-robin linear search strategy where
# one dimension is swept at a time while holding the others at their
# current best value, then re-sweeps earlier dimensions with updated
# optimal values. Generates per-dimension plots, training curves, a
# summary comparison figure, and a CSV file with all results.

# =========================================================================
# EXPERIMENT PLAN
# =========================================================================
# Dataset: Fashion MNIST (more challenging than MNIST digits)
# Base config: patch_size=4, stride=2, embed_dim=48, depth=4,
#              num_heads=8, mlp_dim=128, dropout=0.1, epochs=10
#
# Dimension 1 - Transformer Depth: [1, 2, 4, 6, 8]
#   Hypothesis: Deeper networks will improve accuracy up to a point,
#   then plateau or degrade due to overfitting on this small dataset.
#   Training time should scale roughly linearly with depth.
#
# Dimension 2 - Embedding Dimension: [24, 48, 96, 128, 192]
#   Hypothesis: Larger embeddings will improve accuracy by giving the
#   model more representational capacity, but with diminishing returns.
#   Parameter count and training time will grow significantly.
#
# Dimension 3 - Number of Attention Heads: [1, 2, 4, 8, 12]
#   Hypothesis: More heads allow the model to attend to different
#   features simultaneously. Performance should improve then plateau.
#   Training time impact should be modest since total dimension is fixed.
#
# Strategy: Round-robin linear search
#   Round 1: Sweep depth (hold embed_dim=48, num_heads=8) -> find best depth
#   Round 2: Sweep embed_dim (hold depth=best, num_heads=8) -> find best embed
#   Round 3: Sweep num_heads (hold depth=best, embed_dim=best) -> find best heads
#   Round 4: Re-sweep depth with updated best embed_dim and num_heads
#   Round 5: Re-sweep embed_dim with updated best depth and num_heads
# =========================================================================

import sys
import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import the transformer model and config from our transformer script
from mnist_transformer import NetTransformer, NetConfig


# Loads the Fashion MNIST training and test datasets with normalization
# using precomputed mean and standard deviation. Fashion MNIST is more
# challenging than digit MNIST, providing more room to observe the
# effects of architectural changes.
#
# Arguments:
# batch_size (int) - number of images per training batch (default 64)
#
# Returns:
# train_loader (DataLoader) - DataLoader for the training set
# test_loader (DataLoader)  - DataLoader for the test set
def load_fashion_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


# Trains the model for one complete pass over the training data.
# Moves each batch to the specified device, computes forward pass,
# NLL loss, backpropagation, and weight updates. Returns the average
# loss and accuracy across the full epoch.
#
# Arguments:
# model (NetTransformer)    - the transformer network to train
# train_loader (DataLoader) - DataLoader providing training batches
# optimizer (torch.optim)   - optimizer managing gradient updates
# device (torch.device)     - device to run computations on (cpu/cuda/mps)
#
# Returns:
# avg_loss (float)  - average NLL loss over all training samples
# accuracy (float)  - percentage of correctly classified training samples
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
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


# Evaluates the model on the test dataset without computing gradients.
# Returns the average loss and accuracy over all test samples.
#
# Arguments:
# model (NetTransformer)   - the transformer network to evaluate
# test_loader (DataLoader) - DataLoader providing test batches
# device (torch.device)    - device to run computations on
#
# Returns:
# avg_loss (float)  - average NLL loss over all test samples
# accuracy (float)  - percentage of correctly classified test samples
def evaluate(model, test_loader, device):
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

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy


# Runs a single experiment by building a transformer model from the
# given configuration, training it for the specified number of epochs,
# and evaluating on the test set after each epoch. Tracks training and
# test losses, accuracies, best test accuracy, total parameters, and
# wall-clock training time. Returns all metrics in a dictionary.
#
# Arguments:
# config (NetConfig)        - configuration object with all hyperparameters
# train_loader (DataLoader) - DataLoader for the training set
# test_loader (DataLoader)  - DataLoader for the test set
# device (torch.device)     - device to run computations on
#
# Returns:
# result (dict) - dictionary containing all metrics and training history
def run_experiment(config, train_loader, test_loader, device):
    torch.manual_seed(config.seed)

    # Build model
    model = NetTransformer(config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)

    # Track metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        best_test_acc = max(best_test_acc, test_acc)

    elapsed = time.time() - start_time

    result = {
        'name': config.name,
        'depth': config.depth,
        'embed_dim': config.embed_dim,
        'num_heads': config.num_heads,
        'mlp_dim': config.mlp_dim,
        'dropout': config.dropout,
        'patch_size': config.patch_size,
        'stride': config.stride,
        'total_params': total_params,
        'best_test_acc': best_test_acc,
        'final_test_acc': test_accs[-1],
        'final_train_acc': train_accs[-1],
        'final_test_loss': test_losses[-1],
        'final_train_loss': train_losses[-1],
        'training_time': elapsed,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }

    return result


# Writes all experiment results to a CSV file with one row per
# experiment containing the configuration parameters, accuracy,
# loss, parameter count, and training time.
#
# Arguments:
# results (list)   - list of result dictionaries from run_experiment
# filename (str)   - output CSV file path (default 'outputs/experiment_results.csv')
def save_results_csv(results, filename='outputs/experiment_results.csv'):
    fieldnames = ['name', 'depth', 'embed_dim', 'num_heads', 'mlp_dim',
                  'dropout', 'patch_size', 'stride', 'total_params',
                  'best_test_acc', 'final_test_acc', 'final_train_acc',
                  'final_test_loss', 'final_train_loss', 'training_time']

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)

    print(f"Results saved to {filename}")


# Plots three panels for one dimension of experimentation: best test
# accuracy, training time, and total parameter count versus the varied
# parameter value. Saves the resulting figure to a PNG file.
#
# Arguments:
# results (list)   - list of result dictionaries for this dimension
# dim_name (str)   - human-readable name of the dimension for labels
# dim_key (str)    - dictionary key to extract the dimension value
# filename (str)   - output file path for the saved plot
def plot_dimension_results(results, dim_name, dim_key, filename):
    values = [r[dim_key] for r in results]
    test_accs = [r['best_test_acc'] for r in results]
    train_times = [r['training_time'] for r in results]
    param_counts = [r['total_params'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy vs dimension
    axes[0].plot(values, test_accs, 'b-o', markersize=8)
    axes[0].set_xlabel(dim_name)
    axes[0].set_ylabel('Best Test Accuracy (%)')
    axes[0].set_title(f'Accuracy vs {dim_name}')
    axes[0].grid(True)

    # Training time vs dimension
    axes[1].plot(values, train_times, 'r-o', markersize=8)
    axes[1].set_xlabel(dim_name)
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title(f'Training Time vs {dim_name}')
    axes[1].grid(True)

    # Parameter count vs dimension
    axes[2].plot(values, param_counts, 'g-o', markersize=8)
    axes[2].set_xlabel(dim_name)
    axes[2].set_ylabel('Total Parameters')
    axes[2].set_title(f'Parameters vs {dim_name}')
    axes[2].grid(True)

    plt.suptitle(f'Effect of {dim_name} on Fashion MNIST Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved to {filename}")


# Plots test accuracy and test loss training curves for all experiments
# within one dimension on a single two-panel figure. Each experiment
# is shown as a separate line, labeled by its parameter value, allowing
# visual comparison of convergence speed and final performance.
#
# Arguments:
# results (list)   - list of result dictionaries for this dimension
# dim_name (str)   - human-readable name of the dimension for legend labels
# dim_key (str)    - dictionary key to extract the dimension value
# filename (str)   - output file path for the saved plot
def plot_training_curves(results, dim_name, dim_key, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        label = f'{dim_name}={r[dim_key]}'
        epochs = range(1, len(r['test_accs']) + 1)
        ax1.plot(epochs, r['test_accs'], '-o', markersize=4, label=label)
        ax2.plot(epochs, r['test_losses'], '-o', markersize=4, label=label)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title(f'Test Accuracy Over Training - Varying {dim_name}')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title(f'Test Loss Over Training - Varying {dim_name}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Training curves saved to {filename}")


# Generates a three-panel summary figure comparing best test accuracy
# and training time across all three experimental dimensions. Each
# panel uses dual y-axes with accuracy in blue on the left and
# training time in red on the right.
#
# Arguments:
# all_results (dict) - dictionary mapping dimension names to result lists
# filename (str)     - output file path for the saved plot
def plot_summary(all_results, filename='outputs/experiment_summary.png'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    dim_configs = [
        ('Depth', 'depth', all_results['depth']),
        ('Embedding Dim', 'embed_dim', all_results['embed_dim']),
        ('Num Heads', 'num_heads', all_results['num_heads']),
    ]

    for ax, (dim_name, dim_key, results) in zip(axes, dim_configs):
        values = [r[dim_key] for r in results]
        accs = [r['best_test_acc'] for r in results]
        times = [r['training_time'] for r in results]

        color = 'tab:blue'
        ax.set_xlabel(dim_name)
        ax.set_ylabel('Best Test Accuracy (%)', color=color)
        ax.plot(values, accs, 'b-o', markersize=8)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Training Time (s)', color=color)
        ax2.plot(values, times, 'r-s', markersize=6, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)

        ax.set_title(f'Effect of {dim_name}')

    plt.suptitle('Experiment Summary: Accuracy vs Training Time Trade-offs', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Summary saved to {filename}")


# Main function that orchestrates the full experiment. Loads Fashion
# MNIST data, then runs five rounds of experiments: sweeping depth,
# embedding dimension, and number of attention heads in sequence,
# followed by re-sweeps of depth and embedding dimension with updated
# optimal values from earlier rounds. Prints a final summary of the
# best configuration found and generates all plots and CSV output.
#
# Arguments:
# argv (list) - command line arguments passed from sys.argv
def main(argv):
    os.makedirs('outputs', exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")

    epochs_per_run = 10
    seed = 42

    train_loader, test_loader = load_fashion_mnist(batch_size=64)

    all_results = {'depth': [], 'embed_dim': [], 'num_heads': []}
    full_results = []

    # Current best values (start with defaults)
    best_depth = 4
    best_embed_dim = 48
    best_num_heads = 8

    depth_values = [1, 2, 4, 6, 8]
    embed_dim_values = [24, 48, 96, 128, 192]
    # Note: num_heads must evenly divide embed_dim
    head_values = [1, 2, 4, 8, 12]

    # =========================================================================
    # ROUND 1: Sweep Depth
    # =========================================================================
    print("=" * 60)
    print("ROUND 1: Sweeping Transformer Depth")
    print(f"Fixed: embed_dim={best_embed_dim}, num_heads={best_num_heads}")
    print("=" * 60)

    depth_results = []
    for depth in depth_values:
        name = f'depth_{depth}_embed_{best_embed_dim}_heads_{best_num_heads}'
        print(f"\n--- Running: {name} ---")

        config = NetConfig(
            name=name,
            dataset='fashion_mnist',
            depth=depth,
            embed_dim=best_embed_dim,
            num_heads=best_num_heads,
            epochs=epochs_per_run,
            seed=seed,
            device=str(device),
        )

        result = run_experiment(config, train_loader, test_loader, device)
        depth_results.append(result)
        full_results.append(result)

        print(f"  Best test accuracy: {result['best_test_acc']:.2f}%")
        print(f"  Training time: {result['training_time']:.1f}s")
        print(f"  Parameters: {result['total_params']:,}")

    # Find best depth
    best_depth_result = max(depth_results, key=lambda r: r['best_test_acc'])
    best_depth = best_depth_result['depth']
    all_results['depth'] = depth_results
    print(f"\n>>> Best depth: {best_depth} "
          f"(accuracy: {best_depth_result['best_test_acc']:.2f}%)")

    # =========================================================================
    # ROUND 2: Sweep Embedding Dimension
    # =========================================================================
    print("\n" + "=" * 60)
    print("ROUND 2: Sweeping Embedding Dimension")
    print(f"Fixed: depth={best_depth}, num_heads will be adjusted to fit")
    print("=" * 60)

    embed_results = []
    for embed_dim in embed_dim_values:
        # Ensure num_heads divides embed_dim evenly
        valid_heads = best_num_heads
        while embed_dim % valid_heads != 0:
            valid_heads -= 1

        name = f'depth_{best_depth}_embed_{embed_dim}_heads_{valid_heads}'
        print(f"\n--- Running: {name} ---")

        config = NetConfig(
            name=name,
            dataset='fashion_mnist',
            depth=best_depth,
            embed_dim=embed_dim,
            num_heads=valid_heads,
            mlp_dim=embed_dim * 2,  # scale MLP proportionally
            epochs=epochs_per_run,
            seed=seed,
            device=str(device),
        )

        result = run_experiment(config, train_loader, test_loader, device)
        embed_results.append(result)
        full_results.append(result)

        print(f"  Best test accuracy: {result['best_test_acc']:.2f}%")
        print(f"  Training time: {result['training_time']:.1f}s")
        print(f"  Parameters: {result['total_params']:,}")

    # Find best embedding dimension
    best_embed_result = max(embed_results, key=lambda r: r['best_test_acc'])
    best_embed_dim = best_embed_result['embed_dim']
    all_results['embed_dim'] = embed_results
    print(f"\n>>> Best embed_dim: {best_embed_dim} "
          f"(accuracy: {best_embed_result['best_test_acc']:.2f}%)")

    # =========================================================================
    # ROUND 3: Sweep Number of Attention Heads
    # =========================================================================
    print("\n" + "=" * 60)
    print("ROUND 3: Sweeping Number of Attention Heads")
    print(f"Fixed: depth={best_depth}, embed_dim={best_embed_dim}")
    print("=" * 60)

    head_results = []
    for num_heads in head_values:
        # Skip if num_heads doesn't divide embed_dim evenly
        if best_embed_dim % num_heads != 0:
            print(f"\n--- Skipping heads={num_heads} "
                  f"(doesn't divide embed_dim={best_embed_dim}) ---")
            continue

        name = f'depth_{best_depth}_embed_{best_embed_dim}_heads_{num_heads}'
        print(f"\n--- Running: {name} ---")

        config = NetConfig(
            name=name,
            dataset='fashion_mnist',
            depth=best_depth,
            embed_dim=best_embed_dim,
            num_heads=num_heads,
            mlp_dim=best_embed_dim * 2,
            epochs=epochs_per_run,
            seed=seed,
            device=str(device),
        )

        result = run_experiment(config, train_loader, test_loader, device)
        head_results.append(result)
        full_results.append(result)

        print(f"  Best test accuracy: {result['best_test_acc']:.2f}%")
        print(f"  Training time: {result['training_time']:.1f}s")
        print(f"  Parameters: {result['total_params']:,}")

    # Find best number of heads
    best_head_result = max(head_results, key=lambda r: r['best_test_acc'])
    best_num_heads = best_head_result['num_heads']
    all_results['num_heads'] = head_results
    print(f"\n>>> Best num_heads: {best_num_heads} "
          f"(accuracy: {best_head_result['best_test_acc']:.2f}%)")

    # =========================================================================
    # ROUND 4: Re-sweep Depth with updated best values
    # =========================================================================
    print("\n" + "=" * 60)
    print("ROUND 4: Re-sweeping Depth with optimized embed_dim and num_heads")
    print(f"Fixed: embed_dim={best_embed_dim}, num_heads={best_num_heads}")
    print("=" * 60)

    depth_results_r2 = []
    for depth in depth_values:
        name = f'r2_depth_{depth}_embed_{best_embed_dim}_heads_{best_num_heads}'
        print(f"\n--- Running: {name} ---")

        config = NetConfig(
            name=name,
            dataset='fashion_mnist',
            depth=depth,
            embed_dim=best_embed_dim,
            num_heads=best_num_heads,
            mlp_dim=best_embed_dim * 2,
            epochs=epochs_per_run,
            seed=seed,
            device=str(device),
        )

        result = run_experiment(config, train_loader, test_loader, device)
        depth_results_r2.append(result)
        full_results.append(result)

        print(f"  Best test accuracy: {result['best_test_acc']:.2f}%")
        print(f"  Training time: {result['training_time']:.1f}s")

    best_depth_r2 = max(depth_results_r2, key=lambda r: r['best_test_acc'])
    best_depth = best_depth_r2['depth']
    print(f"\n>>> Updated best depth: {best_depth} "
          f"(accuracy: {best_depth_r2['best_test_acc']:.2f}%)")

    # =========================================================================
    # ROUND 5: Re-sweep Embedding Dimension with updated best values
    # =========================================================================
    print("\n" + "=" * 60)
    print("ROUND 5: Re-sweeping Embedding Dimension")
    print(f"Fixed: depth={best_depth}, num_heads={best_num_heads}")
    print("=" * 60)

    embed_results_r2 = []
    for embed_dim in embed_dim_values:
        valid_heads = best_num_heads
        while embed_dim % valid_heads != 0:
            valid_heads -= 1

        name = f'r2_depth_{best_depth}_embed_{embed_dim}_heads_{valid_heads}'
        print(f"\n--- Running: {name} ---")

        config = NetConfig(
            name=name,
            dataset='fashion_mnist',
            depth=best_depth,
            embed_dim=embed_dim,
            num_heads=valid_heads,
            mlp_dim=embed_dim * 2,
            epochs=epochs_per_run,
            seed=seed,
            device=str(device),
        )

        result = run_experiment(config, train_loader, test_loader, device)
        embed_results_r2.append(result)
        full_results.append(result)

        print(f"  Best test accuracy: {result['best_test_acc']:.2f}%")
        print(f"  Training time: {result['training_time']:.1f}s")

    best_embed_r2 = max(embed_results_r2, key=lambda r: r['best_test_acc'])
    best_embed_dim = best_embed_r2['embed_dim']
    print(f"\n>>> Updated best embed_dim: {best_embed_dim} "
          f"(accuracy: {best_embed_r2['best_test_acc']:.2f}%)")

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Depth: {best_depth}")
    print(f"Best Embedding Dim: {best_embed_dim}")
    print(f"Best Num Heads: {best_num_heads}")
    print(f"Total experiments run: {len(full_results)}")

    # Save all results to CSV
    save_results_csv(full_results)

    # Generate plots for each dimension (using round 1 results for clean comparison)
    print("\nGenerating plots...")

    plot_dimension_results(all_results['depth'], 'Transformer Depth', 'depth',
                           'outputs/exp_depth_results.png')
    plot_training_curves(all_results['depth'], 'Depth', 'depth',
                         'outputs/exp_depth_curves.png')

    plot_dimension_results(all_results['embed_dim'], 'Embedding Dimension', 'embed_dim',
                           'outputs/exp_embed_results.png')
    plot_training_curves(all_results['embed_dim'], 'Embed Dim', 'embed_dim',
                         'outputs/exp_embed_curves.png')

    plot_dimension_results(all_results['num_heads'], 'Number of Heads', 'num_heads',
                           'outputs/exp_heads_results.png')
    plot_training_curves(all_results['num_heads'], 'Heads', 'num_heads',
                         'outputs/exp_heads_curves.png')

    # Summary comparison plot
    plot_summary(all_results)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main(sys.argv)