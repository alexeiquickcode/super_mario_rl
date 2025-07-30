import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import (
    ModelConfig,
    TrainingConfig,
)
from environment import (
    create_train_env,
    get_action_space,
)
from utils import (
    is_remote_fs,
    load_model_checkpoint,
    save_model_checkpoint,
)
from utils.logs import logger_manager

from ..agent import PPOAgent
from .parallel_envs import MultiprocessingEnvWrapper


class Trainer:
    """PPO Trainer for Super Mario Bros."""

    def __init__(
        self,
        training_config: TrainingConfig,
        render: bool = False,
        use_multiprocessing: bool = True,
    ):
        self.training_config = training_config
        self.render = render
        self.use_multiprocessing = use_multiprocessing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get logger for this specific level
        self.logger = logger_manager.get_level_logger(
            self.training_config.world,
            self.training_config.stage,
        )

        # Create agent
        actions: list[list[str]] = get_action_space(self.training_config.action_type)
        num_actions: int = len(actions)
        self.model_config = ModelConfig(num_actions=num_actions)
        self.agent = PPOAgent(training_config, self.model_config)

        # Create environments
        if self.use_multiprocessing:
            self.envs = self._create_environments_mp()
        else:
            self.envs = self._create_environments_no_mp()

        # Create separate rendering environment if rendering is enabled
        self.render_env = None
        if self.render:
            self.render_env = create_train_env(
                self.training_config.world, self.training_config.stage, self.training_config.action_type
            )
            self.render_env.reset()
            self.logger.info("Rendering environment created")

        # Setup local dirs
        if not is_remote_fs(self.training_config.model_path):
            os.makedirs(self.training_config.log_path, exist_ok=True)
            os.makedirs(self.training_config.model_path, exist_ok=True)

        # Tensorboard training metrics tracking
        self.logger.info(f"TensorBoard log directory: {self.training_config.log_path}")
        self.writer = SummaryWriter(log_dir=self.training_config.log_path)
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

    # ---- Environments ----------------------------------------------

    def _create_environments_no_mp(self) -> list:
        """Create environments without multiprocessing."""
        self.logger.info(f"Creating {self.training_config.num_envs} environments (with no mulitprocessing)")
        envs = []
        for i in range(self.training_config.num_envs):
            env = create_train_env(
                self.training_config.world,
                self.training_config.stage,
                self.training_config.action_type,
            )
            envs.append(env)
        return envs

    def _create_environments_mp(self):
        """Create environments with multiprocessing."""
        self.logger.info(f"Creating {self.training_config.num_envs} environments (with multiprocessing)")
        mp_wrapper = MultiprocessingEnvWrapper(
            self.training_config.num_envs, self.training_config.world, self.training_config.stage,
            self.training_config.action_type
        )
        return mp_wrapper

    def _reset_environments(self) -> torch.Tensor:
        """Reset all environments and return initial states."""
        if isinstance(self.envs, MultiprocessingEnvWrapper):
            states = self.envs.reset()
        else:
            states = []
            for env in self.envs:
                state = env.reset()
                states.append(state)
            states = np.stack(states, axis=0)

        states = torch.from_numpy(states).float().to(self.device)
        return states

    def _step_environments(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Step all environments with given actions."""

        actions_np = actions.cpu().numpy()

        if isinstance(self.envs, MultiprocessingEnvWrapper):
            states, rewards, dones, infos = self.envs.step(actions_np)
        else:
            states, rewards, dones, infos = [], [], [], []
            for i, env in enumerate(self.envs):
                action = int(actions_np[i])
                state, reward, done, info = env.step(action)
                states.append(state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            states = np.stack(states, axis=0)

        # Handle rendering environment
        if self.render and self.render_env is not None:
            render_action = actions_np[0]  # Use action from 1st env
            # print(f"Reward: {reward[0]}")
            try:
                self.render_env.render()
                render_state, _, render_done, _ = self.render_env.step(render_action)
                if render_done:
                    self.render_env.reset()
            except Exception as e:
                self.logger.warning(f"Rendering error: {e}")

        states = torch.from_numpy(states).float().to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        return states, rewards, dones, infos

    def collect_rollout(self) -> dict[str, torch.Tensor]:
        """Collect a rollout of experiences from all environments."""

        states_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []

        # Episode tracking for each environment
        episode_rewards = [0.0] * self.training_config.num_envs
        episode_lengths = [0] * self.training_config.num_envs
        completed_episodes = []

        # Reset environments
        current_states: torch.Tensor = self._reset_environments()
        if self.render_env is not None:
            self.render_env.reset()

        # Collect experiences
        for step in range(self.training_config.num_local_steps):

            # 1. Get actions from agent
            actions, log_probs, values = self.agent.get_action(current_states)

            # Store current step data
            states_list.append(current_states)
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            values_list.append(values)

            # 2. Step environments
            next_states, rewards, dones, infos = self._step_environments(actions)

            # Update episode tracking
            for i in range(self.training_config.num_envs):
                episode_rewards[i] += rewards[i].item()
                episode_lengths[i] += 1

                if dones[i]:
                    completed_episodes.append(
                        {
                            'reward': episode_rewards[i],
                            'length': episode_lengths[i],
                            'info': infos[i]
                        }
                    )
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

            # Store step results
            rewards_list.append(rewards)
            dones_list.append(dones)

            # Update current states
            current_states = next_states
            self.total_steps += self.training_config.num_envs

        # Get final value estimate
        with torch.no_grad():
            _, final_values = self.agent.policy(current_states)
            final_values = final_values.squeeze()

        # Compute returns
        returns: torch.Tensor
        returns = self.agent.compute_gae(rewards_list, values_list, dones_list, final_values)

        # Stack all data
        states: torch.Tensor = torch.stack(states_list).view(-1, *states_list[0].shape[1:])
        actions: torch.Tensor = torch.stack(actions_list).view(-1)
        log_probs: torch.Tensor = torch.stack(log_probs_list).view(-1)
        values: torch.Tensor = torch.stack(values_list).view(-1)

        # Flatten returns to match other tensors
        returns = returns.view(-1)
        advantages: torch.Tensor = returns - values

        # Store completed episodes for logging
        self.episode_rewards.extend([ep['reward'] for ep in completed_episodes])
        self.episode_lengths.extend([ep['length'] for ep in completed_episodes])

        rollout_data = {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'returns': returns,
            'advantages': advantages,
            'completed_episodes': completed_episodes
        }

        return rollout_data

    # ---- Training --------------------------------------------------

    def train_step(self) -> dict[str, float]:
        """Perform one training step (collect rollout + update)."""

        # 1. Collect rollout (from all envs)
        rollout_data: dict[str, torch.Tensor] = self.collect_rollout()

        # 2. Update agent
        loss_info = self.agent.update(
            rollout_data['states'],
            rollout_data['actions'],
            rollout_data['log_probs'],
            rollout_data['returns'],
            rollout_data['advantages'],
        )

        self._log_metrics(loss_info, rollout_data)
        return loss_info

    def train(self):
        """Main training loop."""

        # Start from the next episode when resuming, or from episode 0 when starting fresh
        start_episode = self.episode + 1 if hasattr(self, '_resumed_from_checkpoint') else self.episode
        for episode in range(start_episode, self.training_config.num_episodes):
            self.episode = episode
            loss_info = self.train_step()

            # Log
            if episode % 10 == 0:
                recent_rewards = self.episode_rewards[-20:] if self.episode_rewards else [0]
                mean_reward = np.mean(recent_rewards)

                self.logger.info(
                    f"Episode {episode}: "
                    f"Total Loss: {loss_info['total_loss']:.4f}, "
                    f"Actor Loss: {loss_info['actor_loss']:.4f}, "
                    f"Critic Loss: {loss_info['critic_loss']:.4f}, "
                    f"Mean Reward (last 20): {mean_reward:.2f}, "
                    f"Total Steps: {self.total_steps}"
                )

            if episode % self.training_config.save_interval == 0:
                self.save_model(f"episode_{episode}")

        self.writer.close()
        self.close()

    # ---- Utils -----------------------------------------------------

    def _log_metrics(self, loss_info: dict[str, float], rollout_data: dict):
        """Log metrics to tensorboard."""

        # losses
        for loss_name, loss_value in loss_info.items():
            self.writer.add_scalar(f'losses/{loss_name}', loss_value, self.episode)

        # Episode metrics
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
            mean_length = np.mean(self.episode_lengths[-100:])

            self.writer.add_scalar('episode/mean_reward', mean_reward, self.episode)
            self.writer.add_scalar('episode/mean_length', mean_length, self.episode)
            self.writer.add_scalar('episode/total_episodes_completed', len(self.episode_rewards), self.episode)

        # Training metrics
        self.writer.add_scalar('training/total_steps', self.total_steps, self.episode)

        # Value and advantage statistics
        if 'returns' in rollout_data:
            self.writer.add_scalar('training/mean_return', rollout_data['returns'].mean().item(), self.episode)
            self.writer.add_scalar('training/mean_advantage', rollout_data['advantages'].mean().item(), self.episode)
            self.writer.add_scalar('training/advantage_std', rollout_data['advantages'].std().item(), self.episode)

        self.writer.flush()  # Flush for VertexAI TensorBoard real-time viewing

    def close(self):
        """Clean up resources including multiprocessing environments."""
        if isinstance(self.envs, MultiprocessingEnvWrapper):
            self.logger.info("Closing multiprocessing environments...")
            self.envs.close()

        if self.render_env is not None:
            try:
                self.render_env.close()
            except Exception as e:
                self.logger.warning(f"Error closing render environment: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.close()
        except Exception as e:
            # Ignore errors during cleanup
            self.logger.error(f"Error during cleanup: {e}")

    def save_model(self, name: str):
        """Save the current model."""

        model_name = f"ppo_mario_{self.training_config.world}_{self.training_config.stage}_{name}.pt"
        path = f"{self.training_config.model_path}/{model_name}"

        metadata = {
            'action_space': self.training_config.action_type,
            'world': self.training_config.world,
            'stage': self.training_config.stage,
            'episode': self.episode,
            'total_steps': self.total_steps,
        }

        save_model_checkpoint(
            path,
            self.agent.policy,
            self.agent.optimizer,
            self.agent.training_config,
            self.agent.model_config,
            metadata,
        )
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> tuple:
        """Load a model and return checkpoint information."""
        training_config, model_config, metadata = load_model_checkpoint(
            path,
            self.agent.policy,
            self.agent.optimizer,
        )
        self.logger.info(f"Model loaded from {path}")
        return training_config, model_config, metadata

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")

        # Load the checkpoint
        training_config, model_config, metadata = self.load_model(checkpoint_path)

        # Resume from the saved episode and step count
        self.episode = metadata.get('episode', 0)
        self._resumed_from_checkpoint = True  # To calculate remaining episodes
        self.logger.info(f"Resuming from episode {self.episode}")

        # Update training config with remaining episodes (accounting for starting at next episode)
        remaining_episodes = self.training_config.num_episodes - (self.episode + 1)
        if remaining_episodes > 0:
            self.logger.info(f"Will train for {remaining_episodes} more episodes")
        else:
            self.logger.warning(
                f"Checkpoint episode ({self.episode}) >= target episodes ({self.training_config.num_episodes})"
            )
