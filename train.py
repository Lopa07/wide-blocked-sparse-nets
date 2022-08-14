"""Train wide blocked sparse nets.
"""


import argparse
import datetime
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import DATASET
from sparsity import *
from utils import *

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """This function parses the command-line arguments and returns necessary
    parameter values.

    Returns:
        argparse.Namespace: Configuration file with required parameters to train
                            wide blocked sparse nets in PyTorch:
            config_file (str): Configuration yaml file path. Default
                               "config.example.yml"
    """

    parser = argparse.ArgumentParser("Train wide blocked sparse nets in PyTorch.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.example.yml",
        help="Configuration yaml file path",
    )
    return parser.parse_args()


def main(config_file: str) -> None:
    """Train wide blocked sparse nets in PyTorch.

    Args:
        config_file (str): Configuration yaml file path. Default
                           'config.example.yml'
    """

    # Load configuration
    config = yaml.safe_load(Path(config_file).read_text())

    # Logger
    global log_dir
    log_dir = setup_log_dir_and_logger(config_file, config)

    # Tensorboard summary writer
    global tb
    tb = SummaryWriter(log_dir=log_dir)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Set random seed
    set_seed(config["seed"])

    # Dataset
    dataset_name = config["dataset"]["name"]
    dataset = DATASET(
        dataset_name,
        config["training"]["batch_size"]["train"],
        config["training"]["batch_size"]["val"],
    )
    train_loader, val_loader = dataset.load()
    logger.info(f"{dataset_name} training and validation datasets are loaded.")

    # # of data / input channels
    in_channels = dataset.in_channels
    logger.info(f"# of input channels: {in_channels}")

    # # of classes
    num_classes = dataset.num_classes
    logger.info(f"# of classes: {num_classes}")

    # Model and sparse mask
    model_name = config["model"]["name"]
    model, sparse_mask = get_model_and_sparse_mask(
        config["sparsity"],
        model_name,
        in_channels,
        num_classes,
        device,
        logger,
        log_dir,
    )

    # Add model graph to tensorboard
    images, _ = next(iter(val_loader))
    tb.add_graph(model, images)

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config["training"]["optimizer"], model.parameters())
    scheduler = get_scheduler(
        config["training"]["scheduler"], optimizer, config["training"]["num_epochs"]
    )

    # Train
    epochs, train_loss, train_acc, val_loss, val_acc = train(
        config["training"]["num_epochs"],
        model,
        sparse_mask,
        config["sparsity"]["pattern"] == "io_only",
        train_loader,
        val_loader,
        device,
        optimizer,
        criterion,
        scheduler,
    )
    tb.close()

    # Plot loss and accuracy over epochs
    fpath = os.path.join(log_dir, f"train_{model_name}_{dataset_name}")
    plot_loss_accuracy_over_epochs(
        epochs,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        fpath,
    )


def setup_log_dir_and_logger(config_file: str, config: Dict) -> str:
    """Initialize log directory.

    Args:
        config_file (str): Configuration yaml file path
        config (Dict): Run configuration

    Returns:
        str: Log directory
    """

    # Basic run parameters
    dataset = config["dataset"]["name"]
    model = config["model"]["name"]
    base_width = config["sparsity"]["base_width"]
    widening_factor = config["sparsity"]["widening_factor"]
    dist_type = config["sparsity"]["dist_type"]
    pattern = config["sparsity"]["pattern"]

    # Log directory
    log_dir = (
        f"log-{dataset}-{model}-"
        + f"{base_width}-{widening_factor}-{dist_type}-{pattern}"
    )
    log_dir = datetime.datetime.now().strftime(f"{log_dir}-%m_%d_%Y-%H:%M:%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save configuration file for this run
    shutil.copy2(config_file, os.path.join(log_dir, "config.yml"))

    # Logger
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    formatter = logging.Formatter(
        "%(lineno)3s : %(asctime)s : %(name)s : %(levelname)7s : %(message)s"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    return log_dir


def set_seed(seed: int = None) -> None:
    """Initiate training with manual random seed for reproducibility.

    Args:
        seed (int): Random seed for reproducibility
    """

    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.warning(
            "Training with manual random seed. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "This may result in unexpected behavior when restarting from "
            "checkpoints."
        )


def train(
    num_epochs: int,
    model: nn.Module,
    sparse_mask: Dict,
    io_only: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    scheduler: _LRScheduler,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Train model.

    Args:
        num_epochs (int): Number of epochs
        model (nn.Module): Model
        sparse_mask (Dict): Sparsity mask
        io_only (bool): Sparsify convolutional layers along IO dims only
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        criterion (_Loss): Loss function to optimize. Ex. CrossEntropyLoss
        scheduler (_LRScheduler): Learning rate scheduler for optimizer

    Returns:
        List[int]: Training epochs, from start_epoch to start_epoch + num_epochs
        List[float]: Training losses over epochs
        List[float]: Training accuracies over epochs
        List[float]: Validation losses over epochs
        List[float]: Validation accuracies over epochs
    """
    # Start epoch and initial best accuracy
    start_epoch = 0
    best_acc = 0

    # Start training
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    train_start = time.time()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epochs.append(epoch)

        # Training
        loss, acc = train_epoch(
            epoch,
            model,
            sparse_mask,
            io_only,
            train_loader,
            device,
            optimizer,
            criterion,
        )
        train_loss.append(loss)
        train_acc.append(acc)

        # Validation
        loss, acc, best_acc = val_epoch(
            epoch, model, val_loader, device, optimizer, criterion, best_acc, scheduler
        )
        val_loss.append(loss)
        val_acc.append(acc)

        scheduler.step()

    logger.info(f"Training time: {time.time() - train_start:.2f}s")
    return epochs, train_loss, train_acc, val_loss, val_acc


def train_epoch(
    epoch: int,
    model: nn.Module,
    sparse_mask: Dict,
    io_only: bool,
    train_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    criterion: _Loss,
) -> Tuple[float, float]:
    """Train this epoch.

    Args:
        epoch (int): Epoch index
        model (nn.Module): Model
        sparse_mask (Dict): Sparsity mask
        io_only (bool): Sparsify convolutional layers along IO dims only
        train_loader (DataLoader): Training dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        criterion (_Loss): Loss function to optimize. Ex. CrossEntropyLoss

    Returns:
        float: Training loss this epoch
        float: Training accuracy this epoch
    """

    logger.info(f"Training epoch: {epoch}")

    # Initialize
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # Train
    epoch_start = time.time()
    for _, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward to compute gradients
        optimizer.zero_grad()
        loss.backward()

        if sparse_mask:
            # apply sparsity mask to gradients
            for layer, params in model.named_parameters():
                if layer in sparse_mask:
                    mask_tensor(params.grad, sparse_mask[layer], io_only)

        # Optimize
        optimizer.step()

        # Batch train loss and accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = 100 * correct / total

    # Update log
    logger.info(
        f"Loss: {train_loss} | Acc: {train_acc}%, ({correct}/{total}) | "
        f"Time: {time.time() - epoch_start:.2f}s"
    )

    # Add trainig loss, accurary, and weight gradients in tensorboard
    tb.add_scalar("Training loss", train_loss, epoch)
    tb.add_scalar("Training accuracy", train_acc, epoch)

    for name, weight in model.named_parameters():
        tb.add_histogram(name, weight, epoch)
        tb.add_histogram(f"{name}.grad", weight.grad, epoch)

    return train_loss, train_acc


def val_epoch(
    epoch: int,
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    best_acc: float,
    scheduler: _LRScheduler,
) -> Tuple[float, float, float]:
    """Validate this epoch.

    Args:
        epoch (int): Epoch index
        model (nn.Module): Model
        val_loader (DataLoader): Validation dataloader
        device (str): Device being used to train: 'gpu' or 'cpu'
        optimizer (optim.Optimizer): Optimizer. Ex. SGD
        criterion (_Loss): Loss function to optimize. Ex. CrossEntropyLoss
        best_acc (float): Best accuracy before this epoch
        scheduler (_LRScheduler): Learning rate scheduler for optimizer

    Returns:
        float: Validation loss this epoch
        float: Validation accuracy this epoch
        float: Best accuracy after this epoch
    """

    logger.info(f"Validation epoch: {epoch}")

    # Initialize
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    # Validation
    epoch_start = time.time()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Batch validation loss and accuracy
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    # Update log
    logger.info(
        f"Loss: {val_loss} | Acc: {val_acc}%, ({correct}/{total}) | "
        f"Time: {time.time() - epoch_start:.2f}s"
    )

    # Add validation loss and accurary in tensorboard
    tb.add_scalar("Validation loss", val_loss, epoch)
    tb.add_scalar("Validation accuracy", val_acc, epoch)

    # Save checkpoint
    if val_acc >= best_acc:
        logger.info(f"Saving checkpoint from epoch {epoch}.")
        logger.info(f"Best accuracy: was {best_acc}%, now: {val_acc}%.")

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            "acc": val_acc,
        }
        torch.save(state, os.path.join(log_dir, "ckpt.pth"))
        best_acc = val_acc

    return val_loss, val_acc, best_acc


if __name__ == "__main__":
    # Get configuration yaml file path
    args = get_args()

    # Train wide blocked sparse nets in PyTorch
    main(**args.__dict__)
