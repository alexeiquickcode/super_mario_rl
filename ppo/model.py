import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class MarioPolicy(nn.Module):
    """CNN-based policy network for Mario Bros with actor-critic architecture.

    Architecture Diagram:
    ┌─────────────────────────────────┐
    │     Input (N, C, 84, 84)        │
    └─────────────────┬───────────────┘
                      │
    ┌─────────────────▼───────────────┐
    │  Conv1 → (N, 32, 42, 42)        │
    │  ReLU                           │
    └─────────────────┬───────────────┘
                      │
    ┌─────────────────▼───────────────┐
    │  Conv2 → (N, 64, 21, 21)        │
    │  ReLU                           │
    └─────────────────┬───────────────┘
                      │
    ┌─────────────────▼───────────────┐
    │  Conv3 → (N, 128, 11, 11)       │
    │  ReLU                           │
    └─────────────────┬───────────────┘
                      │
    ┌─────────────────▼───────────────┐
    │  Conv4 → (N, 256, 6, 6)         │
    │  ReLU                           │
    └─────────────────┬───────────────┘
                      │
    ┌─────────────────▼───────────────┐
    │  Flatten → (N, 9216)            │
    └─────────────────┬───────────────┘
                      │
    ┌─────────────────▼───────────────┐
    │  Linear → (N, 512)              │
    └─────────────────┬───────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼─────┐            ┌──────▼──┐
    │  Actor   │            │ Critic  │
    │ (N, X)   │            │ (N, 1)  │
    │ [Logits] │            │ [Value] │
    └──────────┘            └─────────┘
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # CNN feature extractor with batch normalization for better GPU utilization
        self.conv1 = nn.Conv2d(config.num_states, config.conv_channels[0], 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(config.conv_channels[0])
        self.conv2 = nn.Conv2d(config.conv_channels[0], config.conv_channels[1], 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(config.conv_channels[1])
        self.conv3 = nn.Conv2d(config.conv_channels[1], config.conv_channels[2], 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(config.conv_channels[2])
        self.conv4 = nn.Conv2d(config.conv_channels[2], config.conv_channels[3], 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(config.conv_channels[3])

        # Calculate flattened size (assumes 84x84 input)
        conv_out_size = config.conv_channels[3] * 6 * 6
        self.linear = nn.Linear(conv_out_size, config.hidden_size)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization

        # Actor-critic heads
        self.critic_head = nn.Linear(config.hidden_size, 1)
        self.actor_head = nn.Linear(config.hidden_size, config.num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, int(nn.init.calculate_gain('relu')))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass returning action logits and value."""

        # CNN feature extraction with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten and pass through MLP with dropout
        x = self.dropout(F.relu(self.linear(x.view(x.size(0), -1))))

        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value
