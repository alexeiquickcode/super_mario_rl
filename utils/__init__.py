import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict, Any


def unwrap_dpp(model: Module) -> Module:
    """
    Unwrap a model from DistributedDataParallel (DDP) if wrapped.

    Parameters
    ----------
    model : torch.nn.Module
        The model instance, possibly wrapped in DistributedDataParallel.

    Returns
    -------
    torch.nn.Module
        The unwrapped model if it was wrapped in DDP; otherwise the original model.
    """
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def save_model_checkpoint(
    filepath: str,
    model: Module,
    optimizer: torch.optim.Optimizer,
    training_config,
    model_config,
    metadata: Optional[Dict[str, Any]] = None,
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
    
    torch.save(checkpoint, filepath)


def load_model_checkpoint(
    filepath: str,
    model: Module,
    optimizer: torch.optim.Optimizer,
    device=None,
) -> tuple:
    """Load model checkpoint with training and model configurations.
    
    Args:
        filepath: Path to checkpoint file.
        model: The model to load state into.
        optimizer: The optimizer to load state into.
        device: Device to map the checkpoint to (optional).
        
    Returns:
        Tuple of (training_config, model_config, metadata) from checkpoint.
        metadata will be an empty dict if not present in checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Handle both old format (policy_state_dict) and new format (model_state_dict)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'policy_state_dict' in checkpoint:
        # Backwards compatibility with old format
        model.load_state_dict(checkpoint['policy_state_dict'])
    else:
        raise KeyError("Checkpoint does not contain 'model_state_dict' or 'policy_state_dict'")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = checkpoint.get('metadata', {})
    return checkpoint['training_config'], checkpoint['model_config'], metadata
