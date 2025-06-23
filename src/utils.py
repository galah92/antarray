import logging
import sys
from pathlib import Path

from jax.experimental.compilation_cache import compilation_cache as cc

# Persistent Jax compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
cc.set_cache_dir("/tmp/jax_cache")


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname} {filename}:{lineno} {message}",
        style="{",
        handlers=[
            logging.FileHandler(Path("app.log"), mode="w+"),  # Overwrite log
            logging.StreamHandler(),
        ],
        force=True,  # https://github.com/google/orbax/issues/1248
    )
    loggers = ("absl", "jax._src.xla_bridge")
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)  # Suppress logging

    # Log the command line arguments
    logger = logging.getLogger(__name__)
    logger.info(f"uv run {' '.join(sys.argv)}")
