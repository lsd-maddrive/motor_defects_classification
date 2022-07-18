import logging
import multiprocessing as mp


def get_n_workers(num_workers: int, logger: logging.Logger) -> int:
    max_cpu_count = mp.cpu_count()
    if num_workers < 0:
        num_workers = max_cpu_count
        logger.info(f"Parameter `num_workers` is set to {num_workers}")

    num_workers = min(max_cpu_count, num_workers)

    return num_workers
