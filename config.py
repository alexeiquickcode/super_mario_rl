import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for PPO training parameters.

    Attributes:
        gamma: GAE reward discount factor, 0.9 is common for short tasks and 
            0.95 - 0.99 for longer horizon tasks.
        tau: GAE smoothing parameter. 1 means no smoothing, if we are getting 
            high variance lets try decreasing it.
        beta: Entropy loss coefficient. This controls exploration larger batches
            tend to overfit faster. We could anneal this over time.
        epsilon: PPO clipping param
    """

    # Environment
    world: int = 1
    stage: int = 1
    action_type: str = "simple"

    # Training hyperparameters
    lr: float = 1e-3
    gamma: float = 0.9
    tau: float = 1.0
    beta: float = 0.01
    epsilon: float = 0.2

    # Training structure
    num_envs: int = 16
    num_local_steps: int = 512
    num_epochs: int = 4
    batch_size: int = 512
    max_grad_norm: float = 0.5

    # Logging and saving
    save_interval: int = 50
    log_path: str = f"{os.getenv("GCS_BUCKET")}/logs"
    model_path: str = f"{os.getenv("GCS_BUCKET")}/models"


@dataclass
class ModelConfig:
    """Configuration for the neural network model."""

    num_states: int = 4  # Frame stack size
    num_actions: int = 7  # Depends on action_type

    # Architecture
    conv_channels: tuple[int, ...] = (64, 128, 256, 512)
    hidden_size: int = 1024
