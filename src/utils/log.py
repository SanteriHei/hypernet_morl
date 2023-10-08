import logging
import logging.config
import pathlib

import tomli

_CONFIG_SET: bool = False

_CONFIG_PATH = pathlib.Path(__file__).parents[2] / "configs" / "log_config.toml"


def get_logger(logger_name: str) -> logging.Logger:
    '''
    Get a new logger (ensures that the configurations are set correctly)

    Parameters
    ----------
    logger_name: str
        The name of the logger. The corresponding file should be found on the 
        configuration file

    Returns
    -------
    logging.Logger: The configured logger.

    '''
    _maybe_set_config(_CONFIG_PATH)
    return logging.getLogger(logger_name)

def info_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs info msg if given condition is True
    
    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.info(msg, stacklevel=2)

def debug_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs debug msg if given condition is True

    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.debug(msg, stacklevel=2)

def warn_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs warning msg if given condition is True

    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool 
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.warning(msg, stacklevel=2)

def critical_if(logger: logging.Logger, cond: bool, msg: str):
    '''
    Logs critical msg if given condition is True

    Parameters
    ----------
    logger: logging.Logger
        The logger used for the logging
    cond: bool
        The condition to check
    msg: str
        The message to log
    '''
    if cond:
        logger.critical(msg, stacklevel=2)

def _maybe_set_config(path: str | pathlib.Path):
    '''
    Set the configuration for the logging lib if not set already.

    Parameters
    ----------
    path: str | pathlib.Path:
        The path to the logging configuration
    '''
    global _CONFIG_SET
    if _CONFIG_SET:
        return

    path = pathlib.Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError((f"{str(path)!r} does not point to a valid "
                                 "logging configuration file!"))

    # rb is required to ensure correct decoding
    with path.open("rb") as ifstream:
        conf = tomli.load(ifstream)
    logging.config.dictConfig(conf)
    _CONFIG_SET = True
