import argparse
import os
from config import TrainingConfig
from ppo.trainer import Trainer
from utils.logs import setup_logger

setup_logger()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Super Mario Bros")

    # Environment args
    action_choices = ["simple", "right", "complex"]
    parser.add_argument("--world", type=int, default=1, help="Mario world (1-8)")
    parser.add_argument("--stage", type=int, default=1, help="Stage number (1-4)")
    parser.add_argument("--action_type", type=str, default="simple", choices=action_choices, help="Action space type")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")

    # Process args
    parser.add_argument("--render", action="store_true", help="Render the first environment during training")
    parser.add_argument("--mp", action="store_true", help="Use multiprocessing or not")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use")

    args = parser.parse_args()

    # Set GPU device if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config = TrainingConfig(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        num_envs=args.num_envs,
        lr=1e-4,
        gamma=0.9,
        num_local_steps=512,
        batch_size=16
    )

    trainer = Trainer(config, render=args.render, use_multiprocessing=args.mp)
    trainer.train(total_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
