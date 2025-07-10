from typing import (
    IO,
    Any,
    TypeVar,
    cast,
)

import fsspec
import torch
import torch.nn as nn
from fsspec.spec import AbstractFileSystem
from torch.nn import Module
from torch.optim import Optimizer

CLOUD_PREFIXES = ('s3://', 'gs://')

T = TypeVar('T', bound=nn.Module)


def get_device() -> torch.device:
    """
    Return the appropriate torch device ('cuda' if available, else 'cpu').

    Returns
    -------
    device : torch.device
        The best available device for computation.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_fs_and_path(path: str) -> tuple[AbstractFileSystem, str]:
    """
    Return the appropriate filesystem handler and cleaned path.

    Parameters
    ----------
    path : str
        The file path or URI to check.

    Returns
    -------
    tuple[AbstractFileSystem, str]
        A tuple containing the filesystem object and the cleaned path.
    """
    if path.startswith(CLOUD_PREFIXES):
        protocol, stripped_path = path.split("://", 1)
        return fsspec.filesystem(protocol), stripped_path
    return fsspec.filesystem("file"), path


def save_model_checkpoint(
    filepath: str,
    model: Module,
    optimizer: Optimizer,
    training_config,
    model_config,
    metadata: dict[str, Any] | None = None,
):
    """Save model checkpoint with training and model configurations.
    
    Args:
        filepath: Path to save checkpoint.
        model: The model to save.
        optimizer: The optimizer to save.
        training_config: Training configuration.
        model_config: Model configuration.
        metadata: Additional metadata to save (e.g., action_space, training_run_info).
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_config': training_config,
        'model_config': model_config,
        'metadata': metadata or {}
    }

    fs, path = get_fs_and_path(filepath)
    with cast(IO[bytes], fs.open(path, 'wb')) as f:
        torch.save(checkpoint, f)
    return


def load_model_checkpoint(
    filepath: str,
    model: Module,
    optimizer: Optimizer,
) -> tuple:
    """Load model checkpoint with training and model configurations.
    
    Args:
        filepath: Path to checkpoint file.
        model: The model to load state into.
        optimizer: The optimizer to load state into.
        
    Returns:
        Tuple of (training_config, model_config, metadata) from checkpoint.
        metadata will be an empty dict if not present in checkpoint.
    """
    fs, path = get_fs_and_path(filepath)
    with cast(IO[bytes], fs.open(path, 'rb')) as f:
        checkpoint = torch.load(f, map_location=get_device(), weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    metadata = checkpoint.get('metadata', {})
    return checkpoint['training_config'], checkpoint['model_config'], metadata


def is_remote_fs(path: str) -> bool:
    return path.startswith(CLOUD_PREFIXES)
