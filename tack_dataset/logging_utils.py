import os
import re
import logging
from datetime import datetime


def setup_logging(log_dir, log_file=None, log_base_name=None, verbose=0):
    """
    Configure logging for the scraping system.
    
    Args:
        log_file: Optional specific log file path. If None, uses timestamped name.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename
    if log_file is None:
        log_base_name = log_base_name or 'log'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{log_base_name}_{timestamp}.log')
        
    # Set log level based on verbosity
    if verbose >= 2:
        log_level = logging.DEBUG
    elif verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler()
        ],
        force=True,
    )

    return log_file


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)