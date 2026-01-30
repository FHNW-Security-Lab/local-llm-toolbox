"""Run the dashboard server."""

import argparse
import logging
import os
import sys

import uvicorn

from .config import config


def setup_logging(debug: bool = False):
    """Configure logging for the application.

    Args:
        debug: If True, enable DEBUG level for our modules (dashboard, manager)
    """
    # Create formatter - more detailed for debug
    if debug:
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(handler)

    # Set levels for our modules
    our_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("dashboard").setLevel(our_level)
    logging.getLogger("dashboard.app").setLevel(our_level)
    logging.getLogger("manager").setLevel(our_level)
    logging.getLogger("manager.control").setLevel(our_level)
    logging.getLogger("manager.tasks").setLevel(our_level)
    logging.getLogger("manager.bridge").setLevel(our_level)
    logging.getLogger("manager.backends").setLevel(our_level)
    logging.getLogger("manager.backends.llama").setLevel(our_level)
    logging.getLogger("manager.backends.foundry").setLevel(our_level)

    # Quiet down noisy libraries (even in debug mode)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("foundry_local").setLevel(logging.WARNING)
    logging.getLogger("sse_starlette").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Local LLM Toolbox")
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging for dashboard and manager modules",
    )
    args = parser.parse_args()

    # Also check environment variable
    debug = args.debug or os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

    setup_logging(debug=debug)

    logger = logging.getLogger("dashboard")
    if debug:
        logger.info("Debug logging enabled")
    logger.info(f"Starting Local LLM Toolbox on http://localhost:{config.port}")

    # sse-starlette handles graceful shutdown properly, so we can use a reasonable timeout
    uvicorn.run(
        "dashboard.app:app",
        host=config.host,
        port=config.port,
        log_level="warning",
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
