import glob
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

from .logs import logger_manager

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
    model: Module | None = None,
    optimizer: Optimizer | None = None,
) -> tuple:
    """Load model checkpoint with training and model configurations.

    Args:
        filepath: Path to checkpoint file.
        model: The model to load state into (optional).
        optimizer: The optimizer to load state into (optional).

    Returns:
        Tuple of (training_config, model_config, metadata) from checkpoint.
        metadata will be an empty dict if not present in checkpoint.
    """
    fs, path = get_fs_and_path(filepath)
    with cast(IO[bytes], fs.open(path, 'rb')) as f:
        checkpoint = torch.load(f, map_location=get_device(), weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = checkpoint.get('metadata', {})
    return checkpoint['training_config'], checkpoint['model_config'], metadata


def is_remote_fs(path: str) -> bool:
    return path.startswith(CLOUD_PREFIXES)


def find_latest_checkpoint(world: int, stage: int, model_path: str) -> str | None:
    """Find the latest checkpoint for a given world and stage, works with both local and GCS paths."""
    try:
        fs, path = get_fs_and_path(model_path)

        # List all files
        try:
            files = fs.ls(path)
        except FileNotFoundError:
            return None

        # Filter for checkpoint files
        checkpoint_files = []
        prefix = f"ppo_mario_{world}_{stage}_episode_"

        for file_info in files:
            if isinstance(file_info, dict):
                filename = file_info['name']
            else:
                filename = file_info

            basename = filename.split('/')[-1]  # Extract filename
            if basename.startswith(prefix) and basename.endswith('.pt'):
                full_path = f"{model_path}/{basename}"
                checkpoint_files.append(full_path)

        if not checkpoint_files:
            return None

        # Sort by episode number (from filename)
        checkpoint_files.sort(key=lambda x: int(x.split('_episode_')[1].split('.')[0]))
        return checkpoint_files[-1]

    except Exception as e:
        logger = logger_manager.get_level_logger(world, stage)
        logger.warning(f"Error searching for checkpoints: {e}")
        return None


def get_latest_model(world: int = 1, stage: int = 1) -> str | None:
    """Get the path to the latest trained model."""

    pattern: str = f"models/ppo_mario_{world}_{stage}_episode_*.pt"
    model_files: list[str] = glob.glob(pathname=pattern)
    if not model_files:
        return None

    # Sort by episode number (from filename)
    model_files.sort(key=lambda x: int(x.split('_episode_')[1].split('.')[0]))
    return model_files[-1]
