import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (
    ModelConfig,
    TrainingConfig,
)

from .model import MarioPolicy


class PPOAgent:
    """PPO Agent that handles policy interactions and training."""

    def __init__(self, training_config: TrainingConfig, model_config: ModelConfig):

        # Main config
        self.training_config = training_config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy network
        self.policy = MarioPolicy(model_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=training_config.lr)

    def get_action(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions from policy for given states.

        Args:
            states: Batch of states [batch_size, channels, height, width]

        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of actions
            values: Value estimates
        """
        # Move states to the same device as the policy model
        states = states.to(self.device)

        with torch.no_grad():
            logits, values = self.policy(states)

        policy = F.softmax(logits, dim=1)
        dist = Categorical(policy)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs, values.squeeze()

    def compute_gae(
        self,
        rewards: list[torch.Tensor],
        values: list[torch.Tensor],
        dones: list[torch.Tensor],
        next_value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalised Advantage Estimation (GAE).

        Args:
            rewards: list of reward tensors for each step
            values: List of value estimates for each step  
            dones: List of done flags for each step
            next_value: Value estimate for the final state

        Returns:
            returns: Discounted returns (advantages + values)
        """
        gae = 0
        returns = []

        # Work backwards through the trajectory
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_values = values[i + 1]

            # GAE calculation
            delta = rewards[i] + self.training_config.gamma * next_values * next_non_terminal - values[i]
            gae = delta + self.training_config.gamma * self.training_config.tau * next_non_terminal * gae
            returns.insert(0, gae + values[i])

        return torch.stack(returns)

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute PPO loss components.

        Args:
            states: Batch of states
            actions: Batch of actions taken
            old_log_probs: Log probabilities under old policy
            returns: Discounted returns
            advantages: Advantage estimates

        Returns:
            Dictionary with loss components
        """
        # Get current policy outputs (actor and critic heads)
        logits, values = self.policy(states)

        # Policy loss
        policy = F.softmax(logits, dim=1)
        dist = Categorical(policy)
        new_log_probs = dist.log_prob(
            actions
        )  # We log probs for stability in RL because probabilities can be really small

        # PPO clipped surrogate objective
        # -------------------------------
        # The surrogate objective is PPO's way to update the policy without taking
        # steps that are too large to destabilise training.
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages  # Standard policy gradient loss
        surr2 = torch.clamp(ratio, 1.0 - self.training_config.epsilon, 1.0 + self.training_config.epsilon) * advantages
        actor_loss = -torch.mean(torch.min(surr1, surr2))  # Take the min to avoid large updates

        # Value loss
        critic_loss = F.smooth_l1_loss(values.squeeze(), returns)

        # Entropy loss for exploration
        entropy_loss = torch.mean(dist.entropy())

        # Total loss
        total_loss = (actor_loss + critic_loss - self.training_config.beta * entropy_loss)

        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss
        }

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        """Update the policy using PPO.

        Args:
            states: Collected states
            actions: Collected actions  
            old_log_probs: Log probabilities under collection policy
            returns: Computed returns
            advantages: Computed advantages

        Returns:
            Dictionary with training metrics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training epochs
        total_samples: int = len(states)
        losses = {'total_loss': [], 'actor_loss': [], 'critic_loss': [], 'entropy_loss': []}

        for epoch in range(self.training_config.num_epochs):

            # Shuffle data
            perm: torch.Tensor = torch.randperm(total_samples)

            # Mini-batch training
            for start in range(0, total_samples, self.training_config.batch_size):
                end = start + self.training_config.batch_size
                batch_indices: torch.Tensor = perm[start:end]

                # Get batch data
                batch_states: torch.Tensor = states[batch_indices]
                batch_actions: torch.Tensor = actions[batch_indices]
                batch_old_log_probs: torch.Tensor = old_log_probs[batch_indices]
                batch_returns: torch.Tensor = returns[batch_indices]
                batch_advantages: torch.Tensor = advantages[batch_indices]

                # Compute loss
                loss_dict = self.compute_loss(
                    batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages
                )

                # Backprop
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()

                # Store losses
                for key, value in loss_dict.items():
                    losses[key].append(value.item())

        # Return average losses
        return {key: float(np.mean(values)) for key, values in losses.items()}
