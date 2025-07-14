import argparse

from config import TrainingConfig
from ppo.train.trainer import Trainer
from utils import find_latest_checkpoint
from utils.logs import logger_manager


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Super Mario Bros")

    # Environment args
    action_choices = ["simple", "right", "complex"]
    parser.add_argument("--world", type=int, default=1, help="Mario world (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Stage number (1-4)")
    parser.add_argument("--action_type", type=str, default="simple", choices=action_choices, help="Action space type")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=750, help="Number of training episodes")

    # Process args
    parser.add_argument("--render", action="store_true", help="Render the first environment during training")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing or not")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use")

    args = parser.parse_args()

    # Logger
    logger = logger_manager.get_logger(f"{args.world}-{args.stage}")

    config = TrainingConfig(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        num_envs=args.num_envs,
        lr=1e-4,
        gamma=0.9,
        num_local_steps=512,
        batch_size=16,
        num_episodes=args.num_episodes
    )

    trainer = Trainer(config, render=args.render, use_multiprocessing=args.mp)

    # Always try to auto-resume from the latest checkpoint
    checkpoint_path = find_latest_checkpoint(args.world, args.stage, config.model_path)
    if checkpoint_path:
        logger.info(f"Found existing checkpoint, resuming from: {checkpoint_path}")
        trainer.resume_from_checkpoint(checkpoint_path)
    else:
        logger.info(f"No checkpoint found in {config.model_path}, starting training from scratch")

    trainer.train()


if __name__ == "__main__":
    main()
