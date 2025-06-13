import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys
default_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

def get_logger(name : str,log_dir = 'logs',log_file = default_filename):

    os.makedirs(log_dir,exist_ok=True)
    log_path = os.path.join(log_dir,log_file)

    log_format = logging.Formatter("[%(asctime)s] %(levelname)s - [%(name)s:%(lineno)d]  - %(message)s")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fileHandler = RotatingFileHandler(log_path,maxBytes=5 * 1024 * 1024, backupCount=3)
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(log_format)
        logger.addHandler(consoleHandler)

    return logger