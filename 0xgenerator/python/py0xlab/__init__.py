from argparse import ArgumentParser
import logging as log
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
log.basicConfig(format=FORMAT)
base_parser = lambda: ArgumentParser()