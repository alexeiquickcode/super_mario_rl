import argparse
import glob
import time

import numpy as np
import torch
from loguru import logger

from config import (
    ModelConfig,
    TrainingConfig,
)
from environment import create_train_env
from ppo.agent import PPOAgent
from utils import load_model_checkpoint
from utils.logs import setup_logger

setup_logger()


class Evaluator:

    def __init__(
        self,
        model_path: str,
        world: int = 1,
        stage: int = 1,
    ):
        self.model_path = model_path
        self.world = world
        self.stage = stage
        self.render = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
        self.env = create_train_env(self.world, self.stage, self.action_type)

    def evaluate_single_episode(
        self,
        max_steps: int = 10000,
        step_delay: float = 0.0,
    ) -> dict:
        """Run a single episode and return statistics."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        while steps < max_steps:

            # Render env
            if self.render:
                self.env.render()
                if step_delay > 0:
                    time.sleep(step_delay)

            # 1. Get action from agent
            actions, _, _ = self.agent.get_action(state_tensor)
            action = actions.item()

            # 2. Take step
            next_state, reward, done, info = self.env.step(action)  # type: ignore

            # 3. Update state
            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            total_reward += reward
            steps += 1
            if done:
                break

        # Episode statistics
        stats = {
            'total_reward': total_reward,
            'steps': steps,
            'final_score': info.get('score', 0),  # type: ignore
            'final_x_pos': info.get('x_pos', 0),  # type: ignore
            'flag_get': info.get('flag_get', False),  # type: ignore
            'time_left': info.get('time', 0),  # type: ignore
        }

        return stats

    def evaluate_multiple_episodes(
        self,
        num_episodes: int = 5,
        max_steps: int = 10000,
        step_delay: float = 0.0,
    ) -> dict:
        """Run multiple episodes and return aggregated statistics."""

        episode_stats = []
        for episode in range(num_episodes):
            logger.info(f"\n{30*'-'} Episode {episode + 1}/{num_episodes} {30*'-'}")

            stats = self.evaluate_single_episode(max_steps=max_steps, step_delay=step_delay)
            episode_stats.append(stats)

            logger.info(f"Total Reward: {stats['total_reward']:.2f}")
            logger.info(f"Steps: {stats['steps']}")
            logger.info(f"Final Score: {stats['final_score']}")
            logger.info(f"Final X-pos: {stats['final_x_pos']}")
            logger.info(f"Flag Get: {stats['flag_get']}")
            logger.info(f"Time Left: {stats['time_left']}")

        aggregated = {
            'mean_reward': np.mean([s['total_reward'] for s in episode_stats]),
            'std_reward': np.std([s['total_reward'] for s in episode_stats]),
            'mean_steps': np.mean([s['steps'] for s in episode_stats]),
            'mean_score': np.mean([s['final_score'] for s in episode_stats]),
            'mean_x_pos': np.mean([s['final_x_pos'] for s in episode_stats]),
            'success_rate': np.mean([s['flag_get'] for s in episode_stats]),
            'mean_time_left': np.mean([s['time_left'] for s in episode_stats])
        }

        return aggregated

    # ---- Utils -----------------------------------------------------

    def _load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from: {self.model_path}")

        # Load checkpoint with safe globals to handle config classes
        with torch.serialization.safe_globals([TrainingConfig, ModelConfig]):
            checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get metadata
        training_config: TrainingConfig = checkpoint['training_config']
        model_config: ModelConfig = checkpoint['model_config']
        metadata = checkpoint.get('metadata', {})
        self.action_type = metadata.get('action_space', 'simple')

        # Load in config
        training_config.world = self.world
        training_config.stage = self.stage
        training_config.action_type = self.action_type

        # Create agent
        self.agent = PPOAgent(training_config, model_config)

        # Load weights
        load_model_checkpoint(self.model_path, self.agent.policy, self.agent.optimizer, self.device)
        self.agent.policy.eval()
        return

    def close(self):
        """Close the environment."""
        self.env.close()


def get_latest_model(world: int = 1, stage: int = 1) -> str | None:
    """Get the path to the latest trained model."""

    pattern = f"models/ppo_mario_{world}_{stage}_episode_*.pt"
    model_files = glob.glob(pattern)
    if not model_files:
        return None

    # Sort by episode number (from filename)
    model_files.sort(key=lambda x: int(x.split('_episode_')[1].split('.')[0]))
    return model_files[-1]


# ------------------------------------------------------------------------------
# ---- Evalution ---------------------------------------------------------------
# ------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO Mario agent")

    # Env args
    parser.add_argument("--world", type=int, default=1, help="Mario world (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Stage number (1-4)")

    # Eval args
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument(
        "--step_delay", type=float, default=0.05, help="Delay between steps (seconds) for better visualization"
    )

    # Get model name
    args = parser.parse_args()
    model_path: str | None = get_latest_model(args.world, args.stage)
    if model_path is None:
        raise ValueError(f"No trained models found for World {args.world}-{args.stage}")

    # Run eval
    evaluator = Evaluator(model_path=model_path, world=args.world, stage=args.stage)
    try:
        logger.info(f"\n--- Multiple Episodes Evaluation ---")
        aggregated = evaluator.evaluate_multiple_episodes(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
        )
        logger.info(f"\nAggregated Results over {args.num_episodes} episodes:")
        logger.info(f"  Mean Reward: {aggregated['mean_reward']:.2f} ± {aggregated['std_reward']:.2f}")
        logger.info(f"  Mean Steps: {aggregated['mean_steps']:.1f}")
        logger.info(f"  Mean Score: {aggregated['mean_score']:.1f}")
        logger.info(f"  Mean X-pos: {aggregated['mean_x_pos']:.1f}")
        logger.info(f"  Success Rate: {aggregated['success_rate']:.2%}")
        logger.info(f"  Mean Time Left: {aggregated['mean_time_left']:.1f}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
