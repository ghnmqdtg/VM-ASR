import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name="", load_existing=False):
    # Create logger
    logger = logging.getLogger(name)
    # If the logger already has handlers, assume it's already configured
    if load_existing and logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    # Remove all handlers that might have been added previously
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(console_handler)

    # Create file handlers
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger
