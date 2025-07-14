import multiprocessing as mp
import queue
from multiprocessing import (
    Process,
    Queue,
)

import numpy as np

from environment import create_train_env

# Set multiprocessing start method to avoid fork issues
mp.set_start_method('spawn', force=True)


def worker(
    worker_id: int,
    env_queue: Queue,
    result_queue: Queue,
    world: int,
    stage: int,
    action_type: str,
) -> None:
    """Worker function that runs in a separate process to handle one environment."""
    try:
        env = create_train_env(world, stage, action_type)
        while True:
            try:
                cmd, data = env_queue.get(timeout=30)  # Get cmd from main process

                if cmd == 'reset':
                    state = env.reset()
                    result_queue.put(('reset', state))
                elif cmd == 'step':
                    action = data
                    state, reward, done, info = env.step(action)
                    result_queue.put(('step', (state, reward, done, info)))
                elif cmd == 'close':
                    env.close()
                    break

            except queue.Empty:
                continue
            except Exception as e:
                result_queue.put(('error', str(e)))
    except Exception as e:
        result_queue.put(('error', str(e)))


class MultiprocessingEnvWrapper:
    """Wrapper to manage multiple environments running in separate processes."""

    def __init__(self, num_envs: int, world: int, stage: int, action_type):
        self.num_envs = num_envs
        self.processes = []
        self.env_queues = []
        self.result_queues = []

        # Create processes and queues
        for i in range(num_envs):
            env_queue = Queue()
            result_queue = Queue()
            process = Process(target=worker, args=(i, env_queue, result_queue, world, stage, action_type))
            process.start()
            self.processes.append(process)
            self.env_queues.append(env_queue)
            self.result_queues.append(result_queue)

    def reset(self):
        """Reset all environments and return initial states."""

        # Send reset commands to all workers
        for env_queue in self.env_queues:
            env_queue.put(('reset', None))

        # Collect results
        states = []
        for result_queue in self.result_queues:
            cmd, state = result_queue.get()
            if cmd == 'error':
                raise RuntimeError(f"Environment reset error: {state}")
            states.append(state)

        return np.stack(states, axis=0)

    def step(self, actions):
        """Step all environments with given actions."""

        # Send step commands to all workers
        for i, env_queue in enumerate(self.env_queues):
            env_queue.put(('step', actions[i]))

        # Collect results
        states, rewards, dones, infos = [], [], [], []
        for result_queue in self.result_queues:
            cmd, result = result_queue.get()
            if cmd == 'error':
                raise RuntimeError(f"Environment step error: {result}")
            state, reward, done, info = result
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return np.stack(states, axis=0), rewards, dones, infos

    def close(self):
        """Close all environments and terminate processes."""

        # Send close commands
        for env_queue in self.env_queues:
            try:
                env_queue.put(('close', None))
            except:
                pass  # Queue might be closed already

        # Wait for processes to finish and terminate if needed
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
