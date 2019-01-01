# Python logger in AWS Lambda has a preset format. To change the format of the logging statement,
# remove the logging handler & add a new handler with the required format

import logging
import sys


def setup_logging():

    logger = logging.getLogger()

    for h in logger.handlers:
      logger.removeHandler(h)

    h = logging.StreamHandler(sys.stdout)

    # use whatever format you want here
    FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

    h.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(h)

    logger.setLevel(logging.INFO)


setup_logging()
