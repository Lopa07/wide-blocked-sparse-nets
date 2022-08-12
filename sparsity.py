"""Helper functions regarding model sparsity:
    - get_model_and_sparse_mask: Initialize model and sparse mask
    - get_model: Model
    - collect_layer_info: Collect layer information
    - get_num_params_to_freeze: Number of parameters to freeze in each layer
    - num_params_to_freeze_w_sparse_type_large_to_small: Number of parameters to
                                                         freeze in each layer.
                                                         Distribute sparsity from
                                                         large to small layers
    - get_sparse_mask: Sparsity mask from # of parameters to freeze in each layer
    - adjust_layer_init: Adjust initial values of weights and biases in sparse 
                         layers
    - get_fan_in: Compute fan-in and "bound" for parameter initialization, for 
                  fully-connected (fc) and convolutional (conv) layers
    - freeze_model_params: Freeze model parameters with sparsity mask
    - mask_tensor: Apply sparsity mask to a given tensor (layer or its gradients)
"""


import bisect
import logging
import os
import random
from collections import defaultdict
from math import ceil
from typing import Dict, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def get_model_and_sparse_mask(
    config: Dict,
    model_name: str,
    in_channels: int,
    num_classes: int,
    device: str,
    logger: logging.Logger,
    log_dir: str,
) -> Tuple[nn.Module, Dict]:
    """Initialize model and sparse mask.

    Args:
        config (Dict): Model sparsity configuration
        model_name (str): Model name to train
        in_channels (int): # of data / input channels
        num_classes (int): # of classes
        device (str): Device being used to train: 'gpu' or 'cpu'
        logger (logging.Logger): Logger
        log_dir (str): Logging directory

    Returns:
        nn.Module: Model to train
        Dict: Sparse masks of all layers
    """

    # Model base width and widened with
    base_width = config["base_width"]
    width = ceil(base_width * config["widening_factor"])

    # Sparse?
    if width < base_width:
        logger.warning(
            f"Model width {width} should be >= {base_width}, the baseline model width. ",
            f"Making model width the same as the base width, {base_width}.",
        )
    logger.info(f"Baseline model width: {base_width}")
    logger.info(f"Training model width: {width}")
    sparse = width > base_width

    # Training model
    model = get_model(model_name, in_channels, num_classes, width, device, logger)

    # Model layer dimensions and parameters
    layers = collect_layer_info(model)

    # Total # of parameters
    num_params_total = sum(layer_num_param for _, _, layer_num_param in layers.values())
    logger.info(f"Total # of paramteers in the training model: {num_params_total}")

    if not sparse:
        return model, None

    # Base model
    base_model = get_model(
        model_name, in_channels, num_classes, base_width, device, logger
    )

    # Base model layer dimensions and parameters
    base_layers = collect_layer_info(base_model)

    # Total # of parameters in the base model
    base_num_params_total = sum(
        layer_num_param for _, _, layer_num_param in base_layers.values()
    )
    logger.info(f"Total # of paramteers in the baseline model: {base_num_params_total}")

    # # of parameters to freeze
    num_params_to_freeze = get_num_params_to_freeze(
        config, layers, num_params_total, base_num_params_total, logger
    )

    # Sparse mask
    sparse_mask = get_sparse_mask(
        layers, num_params_to_freeze, device, config["io_only"]
    )

    # Save sparsity mask
    torch.save(sparse_mask, os.path.join(log_dir, "sparse_mask.pt"))

    # Adjust layer initialization with sparse mask
    adjust_layer_init(model, layers, num_params_to_freeze, logger)

    # Freeze model parameters
    freeze_model_params(model, sparse_mask, config["io_only"], logger)

    return model, sparse_mask


def get_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    width: int,
    device: str,
    logger: logging.Logger,
) -> nn.Module:
    """Model.

    Args:
        model_name (str): Model name to train
        in_channels (int): # of data / input channels
        num_classes (int): # of classes
        width (int): Model width
        device (str): Device being used to train: 'gpu' or 'cpu'
        logger (logging.Logger): Logger

    Returns:
        nn.Moule: Model
    """

    # Model
    model = getattr(__import__("modelzoo"), model_name)(
        in_channels=in_channels, num_classes=num_classes, num_out_conv1=width
    )
    model = model.to(device)
    if device == "cuda":
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    logger.info(f"{model_name} model with width {width} is loaded.")

    return model


def collect_layer_info(model: nn.Module, parent_name: str = "", sep: str = ".") -> Dict:
    """Collect layer information.

    Args:
        model (nn.Module): Model
        parent_name (str): Name of the parent module
        sep (str): Separation in child module name format in state dictionary

    Returns:
        Dict: A dictionary with layer (weight and bias) names, types, dimensions,
              and # of parameters.
    """

    layers = {}
    for layer_name, child in model.named_children():
        layer_name = parent_name + layer_name + sep

        # Leaf
        if hasattr(child, "weight"):
            # Add layer type, weight dimension, and # of parameters
            layers[layer_name + "weight"] = [
                child._get_name(),
                child.weight.shape,
                child.weight.numel(),
            ]

            # Add layer type, bias dimensions, and # of parameters
            if child.bias is not None:
                layers[layer_name + "bias"] = [
                    child._get_name(),
                    child.bias.shape,
                    child.bias.numel(),
                ]

        else:
            layers_child = collect_layer_info(child, layer_name)
            layers.update(layers_child)

    return layers


def get_num_params_to_freeze(
    config: Dict,
    layers: Dict,
    num_params_total: int,
    base_num_params_total: int,
    logger: logging.Logger,
) -> Dict:
    """Number of parameters to freeze in each layer.

    Args:
        config (Dict): Model sparsity configuration
        layers (Dict): Model layer dimensions and # of parameters
        num_params_total (int): Total # of parameters in the model
        base_num_params_total (int): Total # of parameters in the baseline model
        logger (logging.Logger): Logger

    Returns:
        Dict: # of parameters to freeze in each layer
    """
    sparsity_dist_type = config["sparsity_dist_type"]
    if sparsity_dist_type == "large_to_small":
        return num_params_to_freeze_w_sparse_type_large_to_small(
            config, layers, num_params_total, base_num_params_total, logger
        )

    logger.error(f"Sparsity distribution type {sparsity_dist_type} is not supported.")
    exit()


def num_params_to_freeze_w_sparse_type_large_to_small(
    config: Dict,
    layers: Dict,
    num_params_total: int,
    base_num_params_total: int,
    logger: logging.Logger,
) -> Dict:
    """Number of parameters to freeze in each layer. Distribute sparsity from
    large to small layers.

    Args:
        config (Dict): Model sparsity configuration
        layers (Dict): Model layer dimensions and # of parameters
        num_params_total (int): Total # of parameters in the model
        base_num_params_total (int): Total # of parameters in the baseline model
        logger (logging.Logger): Logger

    Returns:
        Dict: # of parameters to freeze in each layer
    """

    # Total # of parameters to freeze
    num_params_to_freeze_total = num_params_total - base_num_params_total

    # Layers to sparsify
    layers = {
        layer: (layer_type, dim, num)
        for layer, (layer_type, dim, num) in layers.items()
        if layer_type in config["layer_types"]
    }
    layers_to_freeze_sorted = sorted(
        layers, reverse=True, key=lambda layer: layers[layer][2]
    )

    # # of layers to freeze
    num_params = [layers[layer][2] for layer in reversed(layers_to_freeze_sorted)]
    num_params_diff = np.diff(num_params)[::-1]

    num_layers = len(num_params)
    num_params_frozen_by_layers = np.cumsum(num_params_diff * np.arange(1, num_layers))
    num_layers_to_freeze = bisect.bisect(
        num_params_frozen_by_layers, num_params_to_freeze_total
    )

    # # of parameters to freeze per layer
    num_params_to_freeze = np.zeros(num_layers, dtype=int)
    base_freeze = np.cumsum(num_params_diff[num_layers_to_freeze - 1 :: -1])[::-1]

    remainder_freeze_total = num_params_to_freeze_total - sum(base_freeze)
    remainder_freeze = remainder_freeze_total // num_layers_to_freeze
    num_params_to_freeze[:num_layers_to_freeze] = base_freeze + remainder_freeze

    if config["io_only"]:
        # If sparsifying convolutional layers along IO dimensions only, # of
        # parameters to freeze in the convolutional layers should be divisible
        # by the kernel size.
        for i in range(num_layers_to_freeze):
            layer = layers_to_freeze_sorted[i]
            tensor_dims = layers[layer][1]
            if len(tensor_dims) == 4:
                kernel_size = tensor_dims[-2] * tensor_dims[-1]
                num_params_to_freeze[i] -= num_params_to_freeze[i] % kernel_size

        remainder = num_params_to_freeze_total - sum(
            num_params_to_freeze[:num_layers_to_freeze]
        )
        for i in range(num_layers_to_freeze):
            layer = layers_to_freeze_sorted[i]
            tensor_dims = layers[layer][1]
            kernel_size = (
                tensor_dims[-2] * tensor_dims[-1] if len(tensor_dims) == 4 else 1
            )
            if remainder >= kernel_size or kernel_size / 2 < remainder:
                num_params_to_freeze[i] += kernel_size
                remainder -= kernel_size

        num_params_freezing = sum(num_params_to_freeze[:num_layers_to_freeze])
        if num_params_freezing + remainder != num_params_to_freeze_total:
            logger.error(
                f"# of parameters freezing {num_params_freezing} + remainder ",
                f"{remainder} is different than the total # of parameters to ",
                f"freeze {num_params_to_freeze_total}.",
            )
            exit()

    else:
        num_params_to_freeze[0] += (
            remainder_freeze_total - remainder_freeze * num_layers_to_freeze
        )
        num_params_freezing = sum(num_params_to_freeze[:num_layers_to_freeze])
        if num_params_freezing != num_params_to_freeze_total:
            logger.error(
                f"# of parameters freezing {num_params_freezing} is different than ",
                f"the total # of parameters to freeze {num_params_to_freeze_total}.",
            )
            exit()

    num_params_to_freeze_dict = defaultdict(int)
    for i, num in enumerate(num_params_to_freeze):
        layer = layers_to_freeze_sorted[i]
        num_params_to_freeze_dict[layer] = num

    return num_params_to_freeze_dict


def get_sparse_mask(
    layers: Dict, num_params_to_freeze: Dict, device: str, io_only: bool
) -> Dict:
    """Sparsity mask from # of parameters to freeze in each layer.

    Args:
        layers (Dict): Model layer dimensions and # of parameters
        num_params_to_freeze (Dict): # of parameters to freeze in each layer
        device (str): Device being used to train: 'gpu' or 'cpu'
        io_only (bool): Sparsify convolutional layers along IO dims only

    Returns:
        Dict: Sparsity mask
    """
    sparse_mask = {}
    for layer, num_params_to_freeze_layer in num_params_to_freeze.items():
        if num_params_to_freeze_layer == 0:
            # Not making this layer sparse
            continue

        # Layer dimensions and # of parameters
        _, tensor_dims, num_params = layers[layer]

        if io_only:
            # If sparsifying convolutional layers along IO dimensions only, set
            # the sparsity indices only for those dimensions.
            if len(tensor_dims) == 4:
                kernel_size = tensor_dims[-2] * tensor_dims[-1]
                num_params //= kernel_size
                num_params_to_freeze_layer //= kernel_size
            tensor_dims = tensor_dims[:2]

        sparse_mask_layer = torch.BoolTensor(num_params).fill_(0)
        if device == "cuda":
            sparse_mask_layer = sparse_mask_layer.cuda()

        # Randomly generate indices of tensor elements to freeze
        indices_to_freeze = random.sample(range(num_params), num_params_to_freeze_layer)
        sparse_mask_layer[indices_to_freeze] = True
        sparse_mask_layer = sparse_mask_layer.view(tensor_dims)

        sparse_mask[layer] = sparse_mask_layer

    return sparse_mask


def adjust_layer_init(
    model: nn.Module, layers: Dict, num_params_to_freeze: Dict, logger: logging.Logger
):
    """Adjust initial values of weights and biases in sparse layers.

    Args:
        model (nn.Module): Model
        layers (Dict): Model layer dimensions and # of parameters
        num_params_to_freeze (Dict): # of parameters to freeze in each layer
        logger (logging.Logger): Logger
    """
    for layer, num_params_to_freeze_layer in num_params_to_freeze.items():
        if num_params_to_freeze_layer == 0:
            # Layer is not sparse
            continue

        if "bias" in layer:
            # Bias are initialized along corresponding weights
            continue

        # Layer dimensions and # of parameters
        layer_type, tensor_dims, num_params = layers[layer]

        # Connectivity of the spared layer
        connectivity = 1 - num_params_to_freeze_layer / num_params

        # Fan-in and "bound" for parameter initialization
        fan_in, bound = get_fan_in(layer, tensor_dims, connectivity)
        if not fan_in:
            logger.error(f"Can not compute fan-in for unknown layer type {layer_type}.")
            exit()

        # Initialize weights
        model.state_dict()[layer].data.uniform_(-bound, bound)

        # Initialize bias
        layer_bias = layer.replace("weight", "bias")
        if layer_bias in layers:
            model.state_dict()[layer_bias].data.uniform_(-bound, bound)


def get_fan_in(
    layer: str, tensor_dims: torch.Size, connectivity: float
) -> Tuple[float, float]:
    """Compute fan-in and "bound" for parameter initialization, for fully-connected
    (fc) and convolutional (conv) layers.

    Args:
        layer (str): Layer name in model state dictionary
        tensor_dims (torch.Size): Tensor dimensions
        connectivity (float): Connectivity of the spared layer

    Returns:
        float: Fan-in for parameter initialization
        float: bound
    """

    if "conv" in layer or "downsample" in layer or "shortcut" in layer:
        # Convolutional layer
        fan_in = tensor_dims[1] * tensor_dims[2] * tensor_dims[3] * connectivity

    elif "fc" in layer or "cl" in layer or "linear" in layer:
        # Fully-connected layer
        fan_in = tensor_dims[1] * connectivity

    else:
        return None, None

    bound = 1 / np.sqrt(fan_in)
    return fan_in, bound


def freeze_model_params(
    model: nn.Module, sparse_mask: Dict, io_only: bool, logger: logging.Logger
):
    """Freeze model parameters with sparsity mask.

    Args:
        model (nn.Module): Model
        sparse_mask (Dict): Sparsity mask
        io_only (bool): Sparsify convolutional layers along IO dims only
        logger (logging.Logger): Logger
    """
    for layer, sparse_mask_layer in sparse_mask.items():
        # Apply sparsity mask to the layer tensor
        tensor = model.state_dict()[layer]
        mask_tensor(tensor, sparse_mask_layer, io_only, logger)


def mask_tensor(
    tensor: torch.Tensor,
    sparse_mask: torch.Tensor,
    io_only: bool,
    logger: logging.Logger = None,
):
    """Apply sparsity mask to a given tensor (layer or its gradients).

    Args:
        tensor (torch.Tensor): Tensor representing a layer or its gradients
        sparse_mask (torch.Tensor): Sparsity mask
        io_only (bool): Sparsify convolutional layers along IO dims only
        logger (logging.Logger): Logger
    """
    # # of tensor dimensions should be 2 or 4
    if tensor.ndim not in {2, 4}:
        logger.error(
            f"# of tensor dimensions should be 2 or 4, but found {tensor.ndim}"
        )
        exit()

    # If sparsifying convolutional layers along IO dimensions only, need to
    # broadcast layer sparsity mask along kernel dimensions.
    if io_only and tensor.ndim == 4:
        sparse_mask = sparse_mask.unsqueeze(2).unsqueeze(3)

    # Freeze layer parameters
    tensor.masked_fill_(sparse_mask, 0)
