from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for PPO training parameters."""
    world: int = 1
    stage: int = 1
    action_type: str = "simple"

    # Training hyperparameters
    lr: float = 1e-4
    gamma: float = 0.9  # discount factor
    tau: float = 1.0  # GAE parameter
    beta: float = 0.01  # entropy coefficient
    epsilon: float = 0.2  # PPO clipping parameter

    # Training structure
    num_envs: int = 8
    num_local_steps: int = 512
    num_epochs: int = 10
    batch_size: int = 16
    max_grad_norm: float = 0.5

    # Logging and saving
    save_interval: int = 50
    log_path: str = "logs/mario_ppo"
    saved_path: str = "models"


@dataclass
class ModelConfig:
    """Configuration for the neural network model."""

    num_states: int = 4  # frame stack size
    num_actions: int = 7  # depends on action_type

    # Architecture
    conv_channels: list = None
    hidden_size: int = 512

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [32, 32, 32, 32]

