import logging
import os
from logging.handlers import RotatingFileHandler

from src.logging.filters import FrameworkFilter


def setup_logging() -> None:
    """
    Configure structured log file routing for SupportSphere AI.

    Log files:
      application.log  — INFO and above, business workflow only, framework-filtered
      pipeline.log     — INFO and above, LangGraph pipeline node execution
      errors.log       — ERROR and above, exception stack traces
    Console: INFO, framework-filtered
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate outputs on Streamlit reruns
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    fw_filter = FrameworkFilter()

    # 1. Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(fw_filter)

    # 2. File Handlers (application.log, pipeline.log, errors.log)
    try:
        app_handler = RotatingFileHandler(
            os.path.join(log_dir, "application.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(formatter)
        app_handler.addFilter(fw_filter)

        pipeline_handler = RotatingFileHandler(
            os.path.join(log_dir, "pipeline.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        pipeline_handler.setLevel(logging.INFO)
        pipeline_handler.setFormatter(formatter)

        errors_handler = RotatingFileHandler(
            os.path.join(log_dir, "errors.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        errors_handler.setLevel(logging.ERROR)
        errors_handler.setFormatter(formatter)

        root_logger.addHandler(app_handler)
        root_logger.addHandler(pipeline_handler)
        root_logger.addHandler(errors_handler)
    except Exception as e:
        print(f"Warning: Could not initialize log files in '{log_dir}': {e}")
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party frameworks
    for f_name in ["nemoguardrails", "pinecone", "urllib3", "httpx", "google", "langchain", "openai"]:
        logging.getLogger(f_name).setLevel(logging.WARNING)
