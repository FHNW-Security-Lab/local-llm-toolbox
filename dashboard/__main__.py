"""Run the dashboard server."""

import logging
import sys

import uvicorn

from .config import config


def setup_logging():
    """Configure logging for the application."""
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Set levels for our modules
    logging.getLogger("dashboard").setLevel(logging.INFO)
    logging.getLogger("manager").setLevel(logging.INFO)

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main():
    setup_logging()

    logger = logging.getLogger("dashboard")
    logger.info(f"Starting Local LLM Toolbox on http://localhost:{config.port}")

    uvicorn.run(
        "dashboard.app:app",
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
