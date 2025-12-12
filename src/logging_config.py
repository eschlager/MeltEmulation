# -*- coding: utf-8 -*-
"""
Created on 17.07.2024
@author: eschlager
"""

import sys
import logging


def define_root_logger(filename='logfile.txt'):
    logger = logging.getLogger()
    # Close and remove only file handlers
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(filename)-18s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Only add a StreamHandler if none exists yet
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)


def close_new_file_handlers(before_handlers):

    for h in list(logging.root.handlers):
        if h not in before_handlers and isinstance(h, logging.FileHandler):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logging.root.removeHandler(h)

    for name, lg in logging.root.manager.loggerDict.items():
        if isinstance(lg, logging.Logger):
            for h in list(lg.handlers):
                if h not in before_handlers and isinstance(h, logging.FileHandler):
                    try:
                        h.flush()
                        h.close()
                    except Exception:
                        pass
                    lg.removeHandler(h)


def close_all_file_handlers():
    # root handlers
    for h in list(logging.root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.flush(); h.close()
            except Exception:
                pass
            logging.root.removeHandler(h)

    # named loggers
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger):
            for h in list(lg.handlers):
                if isinstance(h, logging.FileHandler):
                    try:
                        h.flush(); h.close()
                    except Exception:
                        pass
                    lg.removeHandler(h)

    logging.shutdown()