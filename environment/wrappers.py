from collections import deque
from typing import Any

import cv2
import numpy as np
from gym import (
    Env,
    Wrapper,
)
from gym.spaces import Box
from numpy.typing import NDArray


def process_frame(frame: np.ndarray | None, size: int = 84) -> NDArray[np.float32]:
    """Process a single frame: convert to grayscale and resize.

    Args:
        frame: Input RGB frame or None
        size: Target size for resizing (default 84, can use 42 for 4x speedup)
    """
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (size, size))[None, :, :] / 255.
        return frame.astype(np.float32)
    else:
        return np.zeros((1, size, size), dtype=np.float32)


class CustomReward(Wrapper):
    """Custom reward wrapper for Super Mario Bros with stage-specific logic."""

    def __init__(self, env: Env, world: int, stage: int):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40  # 40 is marios starting x position
        self.max_x = 40  # Track maximum x position reached
        self.world = world
        self.stage = stage
        self.position_history = deque(maxlen=30)  # Track last 30 positions for velocity calculation
        self.position_history.append(40)  # Initialize with starting position

    def step(self, action):
        reward: float
        done: bool
        info: dict[str, Any]
        state, reward, done, info = self.env.step(action)  # type: ignore[using modified gym version]
        state: NDArray[np.float32] = process_frame(state)
        x_pos = info['x_pos']

        # Score-based reward (normalized)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

        # Penalties / rewards
        reward = self._apply_reward_for_new_territory(reward, x_pos)
        reward = self._apply_velocity_based_stagnation_penalty(reward, x_pos)

        # Completion rewards
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                # Mario dies
                reward -= 50

        # Time-based penalty (encourage faster play)
        time_penalty = max(0, (400 - info["time"]) / 100.0)
        reward -= time_penalty  # The more time left, the less penalty

        # Power-up state
        if not info["status"] in ["small", "tall"]:
            # NOTE: CHECKING IF WE ACTUALLY GET FIREBALLS AS PER THE MEMORY MAP
            print('Power-up state:', info["status"])

        # if info["status"] == "tall": TODO: Why are we not seeing these states
        #     reward += 5
        # elif info["status"] == "fireball":
        #     reward += 10

        self.current_x = info["x_pos"]
        return state, reward / 10, done, info

    def _apply_reward_for_new_territory(self, reward: float, x_pos: float) -> float:
        """Exploration reward for reaching new territory"""
        if x_pos > self.max_x:
            exploration_reward: float = (x_pos - self.max_x) / 5.0
            reward += exploration_reward
            self.max_x = x_pos  # Set new pos reached
        return reward

    def _apply_velocity_based_stagnation_penalty(self, reward: float, x_pos: float) -> float:

        # 2. Velocity based stagnation penalty
        self.position_history.append(x_pos)

        # Calculate directional movement over the last 30 steps
        if len(self.position_history) >= 30:
            recent_positions = list(self.position_history)

            # Calculate forward progress (net movement)
            net_movement = recent_positions[-1] - recent_positions[0]

            # Calculate total movement (activity level)
            total_movement = sum(
                abs(recent_positions[i] - recent_positions[i - 1]) for i in range(1, len(recent_positions))
            )
            avg_activity = total_movement / len(recent_positions)

            # Only penalize if BOTH conditions are true:
            # 1. Very low activity (not moving much at all)
            # 2. No net forward progress (not exploring)
            if avg_activity < 0.2 and net_movement <= 0:
                # Mario is truly stuck (low activity + no forward progress)
                stagnation_penalty = (0.2 - avg_activity) * 3.0
                reward -= stagnation_penalty
            elif avg_activity < 0.1:
                # Extremely low activity regardless of direction
                severe_stagnation_penalty = (0.1 - avg_activity) * 5.0
                reward -= severe_stagnation_penalty
        return reward

    def reset(self):  # type: ignore
        self.curr_score = 0
        self.current_x = 40
        self.max_x = 40
        self.position_history.clear()
        self.position_history.append(40)  # Reset with starting position
        return process_frame(self.env.reset())  # type: ignore


class CustomSkipFrame(Wrapper):
    """Frame skipping wrapper that stacks consecutive frames."""

    def __init__(self, env: Env, skip: int = 4):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action: int):
        total_reward = 0
        last_states = []

        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)  # type: ignore[using modified gym version]
            total_reward += reward

            # Store last half of frames for max pooling this is so that all important
            # visual information is included in the state. E.g if an enemy is off the
            # screen in one frame, but is on the screen in the next frame, it will still
            # be captured in the state.
            if i >= self.skip // 2:
                last_states.append(state)

            if done:
                self.reset()
                return self.states.astype(np.float32), total_reward, done, info

        # Max pooling over last frames to handle flickering
        max_state = np.max(np.concatenate(last_states, 0), 0)

        # Update frame stack
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state

        return self.states.astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        if isinstance(state, tuple):  # To appease type checker
            state = state[0]
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states.astype(np.float32)
