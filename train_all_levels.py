from argparse import ArgumentParser, Namespace
import subprocess
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import torch
from loguru import logger


def generate_all_levels() -> list[tuple[int, int]]:
    """Generate all 32 levels (8 worlds × 4 stages)."""
    levels: list[tuple[int, int]] = []
    for world in range(1, 9):
        for stage in range(1, 5):
            levels.append((world, stage))
    return levels


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
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3 * 3600)
        logger.info(f"Completed World {world}-{stage} on GPU {gpu_id}")
        return world, stage, True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed World {world}-{stage} on GPU {gpu_id}")
        return world, stage, False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout World {world}-{stage} on GPU {gpu_id}")
        return world, stage, False


def assign_levels_to_gpus(levels: list[tuple[int, int]], num_gpus: int):
    """Assign levels to GPUs in round-robin fashion."""
    assignments = {gpu_id: [] for gpu_id in range(num_gpus)}

    for i, (world, stage) in enumerate(levels):
        gpu_id = i % num_gpus
        assignments[gpu_id].append((world, stage))

    return assignments


def train_all_levels(num_gpus, levels_per_gpu: int):
    """Train all levels across multiple GPUs."""

    # Level assignments to GPU
    all_levels: list[tuple[int, int]] = generate_all_levels()
    gpu_assignments = assign_levels_to_gpus(all_levels, num_gpus)

    # Train all levels at once
    completed_levels = []
    failed_levels = []
    max_workers = num_gpus * levels_per_gpu
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

    if failed_levels:
        logger.error(f"Failed levels: {failed_levels}")


def main() -> None:
    parser: ArgumentParser = ArgumentParser(description="Train all Super Mario levels across multiple GPUs")
    parser.add_argument(
        "--levels-per-gpu",
        type=int,
        default=1,
        help="Number of levels to train per GPU"
    )
    args: Namespace = parser.parse_args()

    num_gpus: int = torch.cuda.device_count()
    levels_per_gpu: int = args.levels_per_gpu
    logger.info(f"Using all {num_gpus} available GPUs with {levels_per_gpu} levels per GPU")

    train_all_levels(num_gpus, levels_per_gpu)


if __name__ == "__main__":
    main()
