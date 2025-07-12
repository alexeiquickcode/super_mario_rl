import subprocess
import sys
from argparse import (
    ArgumentParser,
    Namespace,
)
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import torch
from loguru import logger


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
        "--gpu",
        str(gpu_id),
        "--mp",
    ]

    cmd_str = " ".join(cmd)
    logger.info(f"Running command: {cmd_str}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3 * 3600)
        logger.info(f"Completed World {world}-{stage} on GPU {gpu_id}")

        if result.stdout:
            logger.info(f"World {world}-{stage} stdout: {result.stdout[-500:]}")  # Last 500 chars

        return world, stage, True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed World {world}-{stage} on GPU {gpu_id}")
        logger.error(f"Command: {cmd_str}")
        logger.error(f"Return code: {e.returncode}")

        # std ERR
        if e.stderr:
            logger.error(f"World {world}-{stage} stderr: {e.stderr}")
        else:
            logger.error(f"World {world}-{stage}: No stderr output")

        # std OUT
        if e.stdout:
            logger.error(f"World {world}-{stage} stdout: {e.stdout}")
        else:
            logger.error(f"World {world}-{stage}: No stdout output")

        return world, stage, False

    except subprocess.TimeoutExpired as e:
        logger.error(f"Timeout World {world}-{stage} on GPU {gpu_id} after {3 * 3600} seconds")
        logger.error(f"Command: {cmd_str}")

        # Log any output from timed out process
        if e.stdout:
            logger.error(f"World {world}-{stage} timeout stdout: {e.stdout[-1000:]}")  # Last 1000 chars
        if e.stderr:
            logger.error(f"World {world}-{stage} timeout stderr: {e.stderr[-1000:]}")  # Last 1000 chars

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
    """Train all levels across multiple GPUs."""

    # Level assignments to GPU
    all_levels: list[tuple[int, int]] = generate_all_levels()
    gpu_assignments = assign_levels_to_gpus(all_levels, num_gpus)

    logger.info(f"GPU assignments: {gpu_assignments}")

    # Train all levels at once
    completed_levels = []
    failed_levels = []
    max_workers = num_gpus * levels_per_gpu
    logger.info(f"Using ThreadPoolExecutor with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # 1. Submit all jobs
        future_to_level = {}
        for gpu_id, levels in gpu_assignments.items():
            for world, stage in levels:
                future = executor.submit(train_level, world, stage, gpu_id)
                future_to_level[future] = (world, stage, gpu_id)

        # 2. Process completed jobs
        for future in as_completed(future_to_level):
            world, stage, gpu_id = future_to_level[future]
            try:
                result_world, result_stage, success = future.result()
                if success:
                    completed_levels.append((result_world, result_stage))
                else:
                    failed_levels.append((result_world, result_stage))

                # Progress update
                total_done: int = len(completed_levels) + len(failed_levels)
                logger.info(f"Progress: {total_done}/{len(all_levels)} levels completed")

            except Exception as e:
                logger.error(f"Unexpected error for World {world}-{stage}: {e}")
                failed_levels.append((world, stage))

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
    logger.info(f"Using all {num_gpus} available GPUs with {levels_per_gpu} levels per GPU")

    if num_gpus == 0:
        logger.error("No GPUs detected! Cannot proceed with training.")
        sys.exit(1)

    train_all_levels(num_gpus, levels_per_gpu)


if __name__ == "__main__":
    main()
