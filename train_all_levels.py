import multiprocessing
import os
import subprocess
import sys
import threading
import time
from argparse import (
    ArgumentParser,
    Namespace,
)
from collections import deque
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import torch
from loguru import logger

# Set multiprocessing start method to avoid fork issues
multiprocessing.set_start_method('spawn', force=True)


def stream_output(process, world, stage, stream_name):
    """Stream output from a subprocess in real-time."""
    stream = process.stdout if stream_name == "stdout" else process.stderr
    captured_lines = []

    try:
        for line in iter(stream.readline, ''):
            if line:
                # Remove trailing newline and add prefix
                clean_line = line.rstrip('\n\r')
                prefixed_line = f"[World {world}-{stage}] {clean_line}"

                # Print to console in real-time
                if stream_name == "stdout":
                    logger.info(prefixed_line)
                else:
                    logger.error(prefixed_line)

                # Store for result
                captured_lines.append(line)
    except Exception as e:
        logger.error(f"Error streaming {stream_name} for World {world}-{stage}: {e}")
    finally:
        stream.close()

    return ''.join(captured_lines)


def train_level(world: int, stage: int, gpu_id: int):
    """Train a single level on a specific GPU."""
    logger.info(f"Starting World {world}-{stage} on GPU {gpu_id}")

    cmd: list[str] = [
        "python",
        "train.py",
        "--world",
        str(world),
        "--stage",
        str(stage),
        "--mp",
    ]

    # Set up environment with specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Log process command
    cmd_str = " ".join(cmd)
    logger.info(f"Running command: {cmd_str} with GPU {gpu_id}")

    try:
        # Use Popen for real-time streaming
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Start threads to stream output in real-time
        stdout_thread = threading.Thread(target=stream_output, args=(process, world, stage, "stdout"))
        stderr_thread = threading.Thread(target=stream_output, args=(process, world, stage, "stderr"))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Wait for process to complete with timeout
        try:
            return_code = process.wait(timeout=3 * 3600)
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout World {world}-{stage} on GPU {gpu_id} after {3 * 3600} seconds")
            process.kill()
            return world, stage, False

        # Wait for streaming threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        if return_code == 0:
            logger.info(f"Completed World {world}-{stage} on GPU {gpu_id}")
            return world, stage, True
        else:
            logger.error(f"Failed World {world}-{stage} on GPU {gpu_id} with return code {return_code}")
            return world, stage, False

    except Exception as e:
        logger.error(f"Unexpected error World {world}-{stage} on GPU {gpu_id}: {type(e).__name__}: {e}")
        logger.error(f"Command: {cmd_str}")
        return world, stage, False


def assign_levels_to_gpus(levels: list[tuple[int, int]], num_gpus: int):
    """Assign levels to GPUs in round-robin fashion."""
    assignments = {gpu_id: [] for gpu_id in range(num_gpus)}

    for i, (world, stage) in enumerate(levels):
        gpu_id = i % num_gpus
        assignments[gpu_id].append((world, stage))

    return assignments


def generate_all_levels() -> list[tuple[int, int]]:
    """Generate all 32 levels (8 worlds × 4 stages)."""
    levels: list[tuple[int, int]] = []
    for world in range(1, 9):
        for stage in range(1, 5):
            levels.append((world, stage))
    return levels


# ------------------------------------------------------------------------------
# ---- Primary Functions -------------------------------------------------------
# ------------------------------------------------------------------------------


def train_all_levels(num_gpus, levels_per_gpu: int):
    """Train all levels across multiple GPUs with proper scheduling."""

    # Level assignments to GPU
    all_levels: list[tuple[int, int]] = generate_all_levels()
    gpu_assignments = assign_levels_to_gpus(all_levels, num_gpus)

    # Log assignment summary
    logger.info(f"Using {num_gpus} GPUs for training")
    for gpu_id, levels in gpu_assignments.items():
        logger.info(f"GPU {gpu_id}: {len(levels)} levels - {levels}")

    completed_levels = []
    failed_levels = []

    # Use controlled submission to respect levels_per_gpu
    max_workers: int = num_gpus * levels_per_gpu
    logger.info(f"Using ThreadPoolExecutor with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Track job queues and active jobs per GPU
        pending_jobs = {gpu_id: deque(levels) for gpu_id, levels in gpu_assignments.items()}
        active_jobs_per_gpu = {gpu_id: 0 for gpu_id in range(num_gpus)}
        future_to_info = {}

        # 1. Submit initial batch (up to X per GPU)
        for gpu_id in range(num_gpus):
            for _ in range(min(levels_per_gpu, len(pending_jobs[gpu_id]))):
                if pending_jobs[gpu_id]:
                    world, stage = pending_jobs[gpu_id].popleft()
                    future = executor.submit(train_level, world, stage, gpu_id)
                    future_to_info[future] = (world, stage, gpu_id)
                    active_jobs_per_gpu[gpu_id] += 1

                    logger.info(f"Submitted: World {world}-{stage} to GPU {gpu_id}")

                    # NOTE: Small delay to stagger submissions, 10-30s is required
                    # to avoid overloading the CPU initially as we are spinning up
                    # lot of processes at once.
                    time.sleep(15)

        # 2. Process completed jobs and submit new ones
        while future_to_info:

            # Wait for at least one job to complete
            completed_future = next(as_completed(future_to_info))
            world, stage, gpu_id = future_to_info[completed_future]

            result_world, result_stage, success = completed_future.result()
            if success:
                completed_levels.append((result_world, result_stage))
            else:
                failed_levels.append((result_world, result_stage))

            # Remove completed job and update counters
            del future_to_info[completed_future]
            active_jobs_per_gpu[gpu_id] -= 1

            # Progress update
            total_done = len(completed_levels) + len(failed_levels)
            logger.info(f"Progress: {total_done}/{len(all_levels)} levels completed")

            # Submit next job for this GPU if available
            if pending_jobs[gpu_id] and active_jobs_per_gpu[gpu_id] < levels_per_gpu:
                next_world, next_stage = pending_jobs[gpu_id].popleft()
                next_future = executor.submit(train_level, next_world, next_stage, gpu_id)
                future_to_info[next_future] = (next_world, next_stage, gpu_id)
                active_jobs_per_gpu[gpu_id] += 1
                logger.info(f"Submitted: World {next_world}-{next_stage} to GPU {gpu_id}")

    # Final summary
    logger.info("=== Training Summary ===")
    logger.info(f"Total levels: {len(all_levels)}")
    logger.info(f"Completed levels: {len(completed_levels)}")
    logger.info(f"Failed levels: {len(failed_levels)}")

    if completed_levels:
        logger.info(f"Successful levels: {completed_levels}")

    if failed_levels:
        logger.error(f"Failed levels: {failed_levels}")
        logger.error("Some training processes failed")

        try:
            # Log system state after failures
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.error("GPU status after failures:")
                logger.error(result.stdout)
        except Exception as e:
            logger.error(f"Could not check GPU status: {e}")


def main() -> None:
    parser: ArgumentParser = ArgumentParser(description="Train all Super Mario levels across multiple GPUs")
    parser.add_argument("--levels-per-gpu", type=int, default=1, help="Number of levels to train per GPU")
    args: Namespace = parser.parse_args()

    num_gpus: int = torch.cuda.device_count()
    levels_per_gpu: int = args.levels_per_gpu

    logger.info(f"Detected {num_gpus} GPUs")

    if num_gpus == 0:
        logger.error("No GPUs detected! Cannot proceed with training.")
        sys.exit(1)

    logger.info(f"Using all {num_gpus} available GPUs with {levels_per_gpu} levels per GPU")

    train_all_levels(num_gpus, levels_per_gpu)


if __name__ == "__main__":
    main()
