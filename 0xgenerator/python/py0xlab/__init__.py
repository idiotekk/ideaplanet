from argparse import ArgumentParser
import logging as log
from PIL import Image
from tqdm import tqdm
from pathlib import Path


__all__ = [
    "log",
    "Image",
    "base_parser",
    "tqdm",
    "Path",
]

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
log.basicConfig(format=FORMAT, level=log.DEBUG)

base_parser = lambda: ArgumentParser()