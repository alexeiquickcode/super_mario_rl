import gym_super_mario_bros
from gym import Env
from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)
from nes_py.wrappers import JoypadSpace

from .wrappers import (
    CustomReward,
    CustomSkipFrame,
)


def get_action_space(action_type: str) -> list[list[str]]:
    """Get the action space based on action type."""
    if action_type == "right":
        return RIGHT_ONLY
    elif action_type == "simple":
        return SIMPLE_MOVEMENT
    elif action_type == "complex":
        return COMPLEX_MOVEMENT
    else:
        raise ValueError(f"Unknown action type: {action_type}")


def create_train_env(world: int, stage: int, action_type: str) -> Env:
    """Create a training environment for Super Mario Bros.

    Args:
        world: Mario world number (1-8)
        stage: Stage number (1-4)
        action_type: Type of action space ("simple", "right", "complex")

    Returns:
        Wrapped environment ready for training
    """
    # Create base environment
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")

    # Add action space wrapper
    actions = get_action_space(action_type)
    env = JoypadSpace(env, actions)

    # Custom wrappers
    env = CustomReward(env, world, stage)
    env = CustomSkipFrame(env)
    return env

