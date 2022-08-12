"""Helper functions:
    - get_optimizer: Optimizer
    - get_scheduler: Learning rate scheduler
    - compare_configs: Compare configurations
    - plot_loss_accuracy_over_epochs: Plot training and validation loss and
                                      accuracy over epochs
"""


from typing import Dict, Iterator, List

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer(
    optimizer_config: Dict, parameters: Iterator[Parameter]
) -> optim.Optimizer:
    """Optimizer.

    Args:
        optimizer_config (Dict): Dictionary with optimizer configurations.
            optimizer:
                name (str): 'sgd', 'nesterov_sgd', 'rmsprop', 'adagrad', or
                            'adam'
                learning_rate (float): Initial learning rate
                momentum (float): Momentum factor
                weight_decay (float): Weight decay (L2 penalty)
        parameters (Iterator[Parameter]): Model parameters

    Returns:
        optim.Optimizer: Optimizer
    """

    # Optimizer name, learning rate, momentum, and weight decay
    optimizer_name = optimizer_config["name"]
    lr = optimizer_config["learning_rate"]
    momentum = optimizer_config["momentum"]
    wd = optimizer_config["weight_decay"]

    # Optimizer
    if optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)

    if optimizer_name == "nesterov_sgd":
        return optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True
        )

    if optimizer_name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=wd)

    if optimizer_name == "adagrad":
        return optim.Adagrad(parameters, lr=lr, weight_decay=wd)

    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=wd)


def get_scheduler(
    scheduler_config: Dict, optimizer: optim.Optimizer, num_epochs: int
) -> _LRScheduler:
    """Learning rate scheduler.

    Args:
        scheduler_config (Dict): Dictionary with scheduler configurations.
            scheduler:
                name (str): 'constant', 'step', 'multistep', 'exponential', or
                            'cosine'
                kwargs (Dict): Scheduler specific key word arguments
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        num_epochs (int): Number of epochs

    Returns:
        _LRScheduler: Learning rate scheduler for optimizer
    """

    # Scheduler name and kwargs
    scheduler_name = scheduler_config["name"]
    kwargs = scheduler_config["kwargs"]

    # Scheduler
    if scheduler_name == "constant":
        return optim.lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)

    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, **kwargs)

    if scheduler_name == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 120, 200], gamma=0.1
        )

    if scheduler_name == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer, (1e-3) ** (1 / num_epochs), **kwargs
        )

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1
        )


def compare_configs(config: Dict, config_checkpoint: Dict) -> bool:
    """Compare configurations.

    Args:
        config (Dict): Configuration for this run
        config_checkpoint (Dict): Configuration for checkpoint run

    Returns:
        bool: Check if basic configurations are same with the checkpoint and can
              resume training
    """
    return (
        config["model"] == config_checkpoint["model"]
        and config["dataset"] == config_checkpoint["dataset"]
        and config["training"]["optimizer"]["name"]
        == config_checkpoint["training"]["optimizer"]["name"]
        and config["training"]["scheduler"]
        == config_checkpoint["training"]["scheduler"]
    )


def plot_loss_accuracy_over_epochs(
    epochs: List[int],
    train_loss: List[float],
    train_acc: List[float],
    val_loss: List[float],
    val_acc: List[float],
    fpath: str,
):
    """Plot training and validation loss and accuracy over epochs.

    Args:
        epochs (List[int]): Training epochs, from start_epoch to start_epoch +
                            num_epochs
        train_loss (List[float]): Training losses over epochs
        train_acc (List[float]): Training accuracies over epochs
        val_loss (List[float]): Validation losses over epochs
        val_acc (List[float]): Validation accuracies over epochs
        fpath (str): Png file path to save the plot
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

    ax0.plot(epochs, train_loss, label="Train")
    ax0.plot(epochs, val_loss, label="Validation")
    ax0.grid(True)
    ax0.set_ylabel("Loss")

    ax1.plot(epochs, train_acc, label="Train")
    ax1.plot(epochs, val_acc, label="Validation")
    ax1.grid(True)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    lines, labels = ax0.get_legend_handles_labels()
    fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(0.7, 0.45, 0.5, 0.5))

    fig.tight_layout()
    plt.savefig(fpath, bbox_inches="tight")
