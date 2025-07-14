from typing import Any

import cv2
import numpy as np
from gym import (
    Env,
    Wrapper,
)
from gym.spaces import Box
from numpy.typing import NDArray


def process_frame(frame: np.ndarray | None) -> NDArray[np.float32]:
    """Process a single frame: convert to grayscale and resize."""
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame.astype(np.float32)
    else:
        return np.zeros((1, 84, 84), dtype=np.float32)


class CustomReward(Wrapper):
    """Custom reward wrapper for Super Mario Bros with stage-specific logic."""

    def __init__(self, env: Env, world: int, stage: int):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage

    def step(self, action):
        reward: float
        done: bool
        info: dict[str, Any]
        state, reward, done, info = self.env.step(action)  # type: ignore[using modified gym version]
        state: NDArray[np.float32] = process_frame(state)

        # Score-based reward (normalized)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

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
        if not info["status"] in ["small", "talls"]:
            print('Power-up state:', info["status"])

        # if info["status"] == "tall": TODO: Why are we not seeing these states
        #     reward += 5
        # elif info["status"] == "fireball":
        #     reward += 10

        self.current_x = info["x_pos"]
        return state, reward / 10, done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return process_frame(self.env.reset())


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
