import logging
import os


def create_logger(log_root, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = '[%(asctime)s] (%(name)s): %(levelname)s %(message)s'

    file_handler = logging.FileHandler(os.path.join(log_root, f"{name}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    return logger


def main():
    pass


if __name__ == '__main__':
    main()