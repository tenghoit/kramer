import logging
import datetime
from pathlib import Path


def setup_logging(log_name: str | None = None) -> None:
    """
    Set up logging configuration.
    By default, logs are saved in the 'logs' directory with the current date as the filename.
    """
    if log_name is None:
        log_name = datetime.datetime.now().strftime("%Y%m%d.log")
    log_path = Path(__file__).resolve().parents[1] / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s:%(levelname)s:%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path), 
            logging.StreamHandler() # log to console
        ]
    )


setup_logging()
