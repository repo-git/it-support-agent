# src/utils/logging_setup.py
"""
Setup logging configurazione per IT Support Agent
"""
import logging
import logging.handlers
import os
from pathlib import Path
import colorlog


def setup_logging(log_level: str = "INFO"):
    """Configura il sistema di logging con colori e file rotation"""

    # Crea directory logs se non esiste
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configurazione livello log
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Formatter per console con colori
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # Formatter per file (senza colori)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # Handler file con rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "it_support_agent.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)

    # Handler errori separato
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)

    # Configurazione root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Rimuovi handler esistenti
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Aggiungi nuovi handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # Configurazione logger specifici
    logging.getLogger("livekit").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
