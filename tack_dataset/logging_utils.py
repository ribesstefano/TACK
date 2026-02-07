import os
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