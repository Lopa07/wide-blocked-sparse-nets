"""Summarize results from varying base widths and widening factors, for a given
`dataset`, `model`, and sparsity distribution.
"""


import argparse
import glob
import os
from typing import Dict

import matplotlib.pyplot as plt
import torch


def get_args() -> argparse.Namespace:
    """This function parses the command-line arguments and returns necessary
    parameter values.

    Returns:
        argparse.Namespace: Configuration file with required parameters to
                            summarize results from varying base widths and
                            widening factors:
            dataset (str): Dataset name to classify. choices are "CIFAR10",
                           "CIFAR100", "SVHN", "MNIST", and "FashionMNIST".
                           Default "CIFAR10"
            model (str): Model to train on. choices are "ResNet18". Default
                         "ResNet18"
            sparsity_dist_type (str): Sparsity distribution type along layers.
                                      Choices are "large_to_small" and "uniform".
                                      Default "large_to_small"
            sparsity_pattern (str): Sparsity pattern within a layer. Choices are
                                    "random", "io_only", and "blocked". Default
                                    "random"
    """

    parser = argparse.ArgumentParser("Summarize results.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices={"CIFAR10", "CIFAR100", "SVHN", "MNIST", "FashionMNIST"},
        default="CIFAR10",
        help="Dataset to classify",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices={"ResNet18"},
        default="ResNet18",
        help="Model to train on",
    )
    parser.add_argument(
        "-sdt",
        "--sparsity_dist_type",
        choices={"large_to_small", "uniform"},
        type=str,
        default="large_to_small",
        help="Sparsity distribution type along layers",
    )
    parser.add_argument(
        "-sp",
        "--sparsity_pattern",
        choices={"random", "io_only", "blocked"},
        type=str,
        default="random",
        help="Sparsity pattern within a layer",
    )
    return parser.parse_args()


def main(
    dataset: str, model: str, sparsity_dist_type: str, sparsity_pattern: str
) -> None:
    """Summarize results from varying base widths and widening factors.

    Args:
        dataset (str): Dataset name to classify. choices are "CIFAR10",
                       "CIFAR100", "SVHN", "MNIST", and "FashionMNIST"
        model (str): Model to train on. choices are "ResNet18"
        sparsity_dist_type (str): Sparsity distribution type along layers.
                                  Choices are "large_to_small" and "uniform"
        sparsity_pattern (str): Sparsity pattern within a layer. Choices are
                                "random", "io_only", and "blocked"
    """

    # Collect results from varying base widths and widening factors, for a given
    # `dataset`, `model`, and sparsity distribution.
    results = collect_results(dataset, model, sparsity_dist_type, sparsity_pattern)

    # Plot results
    plot_results(results, dataset, model, sparsity_dist_type, sparsity_pattern)


def collect_results(
    dataset: str, model: str, sparsity_dist_type: str, sparsity_pattern: str
) -> Dict:
    """Collect results from varying base widths and widening factors, for a given
    `dataset`, `model`, and sparsity distribution.

    Args:
        dataset (str): Dataset name to classify. choices are "CIFAR10",
                       "CIFAR100", "SVHN", "MNIST", and "FashionMNIST"
        model (str): Model to train on. choices are "ResNet18"
        sparsity_dist_type (str): Sparsity distribution type along layers.
                                  Choices are "large_to_small" and "uniform"
        sparsity_pattern (str): Sparsity pattern within a layer. Choices are
                                "random", "io_only", and "blocked"

    Returns:
        Dict: Best validation accuracies over varying base widths and widening
              factors
    """
    results = {}

    # List of checkpoint directories
    checkpoint_dirs = glob.glob(
        f"log-{dataset}-{model}-*-{sparsity_dist_type}-{sparsity_pattern}-*"
    )

    for checkpoint_dir in checkpoint_dirs:
        # Base width and widening factor
        base_width, widening_factor = checkpoint_dir.split("-")[3:5]
        base_width = int(base_width)
        widening_factor = float(widening_factor)

        # Best validation accuracy
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pth")
        try:
            checkpoint = torch.load(checkpoint_path)
            best_val_acc = checkpoint["acc"]

        except FileNotFoundError:
            print(f"Checkpoint path '{checkpoint_path}' is not present!")
            exit()

        # Update results
        if base_width not in results:
            results[base_width] = {"widening_factor": [], "best_val_acc": []}

        results[base_width]["widening_factor"].append(widening_factor)
        results[base_width]["best_val_acc"].append(best_val_acc)

    # Sort best validation accuracies by widening factors
    for results_width in results.values():
        widening_factor = results_width["widening_factor"]
        best_val_acc = results_width["best_val_acc"]

        idx = sorted(range(len(widening_factor)), key=widening_factor.__getitem__)
        results_width["widening_factor"] = [widening_factor[i] for i in idx]
        results_width["best_val_acc"] = [best_val_acc[i] for i in idx]

    return results


def plot_results(
    results: Dict,
    dataset: str,
    model: str,
    sparsity_dist_type: str,
    sparsity_pattern: str,
) -> None:
    """Plot results.

    Args:
        results (Dict): Best validation accuracies over varying base widths and
                        widening factors
        dataset (str): Dataset name to classify. choices are "CIFAR10",
                       "CIFAR100", "SVHN", "MNIST", and "FashionMNIST"
        model (str): Model to train on. choices are "ResNet18"
        sparsity_dist_type (str): Sparsity distribution type along layers.
                                  Choices are "large_to_small" and "uniform"
        sparsity_pattern (str): Sparsity pattern within a layer. Choices are
                                "random", "io_only", and "blocked"
    """
    colors = ["k", "b", "c", "g", "r"]
    markers = ["^", "x", "o", "s", "d"]

    # Plot
    for i, base_width in enumerate(sorted(results)):
        widening_factor = results[base_width]["widening_factor"]
        best_val_acc = results[base_width]["best_val_acc"]
        plt.plot(
            widening_factor,
            best_val_acc,
            color=colors[i],
            marker=markers[i],
            label=f"base width = {base_width}",
        )

    plt.grid(True, which="both")
    plt.xscale("log")
    plt.legend()

    plt.xlabel("widening factor")
    plt.ylabel("best validation accuracy")
    plt.title(
        f"Dataset: {dataset}, Model: {model}, "
        + f"Sparsity: {sparsity_dist_type} {sparsity_pattern}"
    )
    plt.tight_layout()

    # Save the plot
    fpath = "log-summary"
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    fpath = os.path.join(
        fpath,
        f"log-{dataset}-{model}-{sparsity_dist_type}-{sparsity_pattern}-summary",
    )
    plt.savefig(fpath, bbox_inches="tight")


if __name__ == "__main__":
    # Get command-line arguments
    args = get_args()

    # Summarize and plot results
    main(**args.__dict__)
