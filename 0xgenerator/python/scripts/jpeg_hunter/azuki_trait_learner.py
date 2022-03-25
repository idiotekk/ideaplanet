from asyncio import start_unix_server
from logging import root
import pandas as pd
from glob import glob
import sys, os
import pprint
import numpy as np
from PIL import ImageFilter
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
pprint.pprint(sys.path)
from py0xlab import *
from py0xlab.frame import where, is_same_color, get_bg_color
from py0xlab import frame as frm
from py0xlab import io, algo
from pathlib import Path
from PIL import ImageEnhance, ImageFilter
from tqdm import tqdm
from argparse import ArgumentParser
from math import pi
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

#parser = ArgumentParser()
#parser.add_argument("file_name", type=str)

slug = "azuki"
root_dir = Path("/Users/zche/data/0xgenerator/database/") / slug
im_dir = root_dir / slug
info_file = root_dir / "trait_info" / f"{slug}.csv"

def read_asset(num, **kw):
    return io.read_frame(root_dir / f"{num}.png", **kw)

    
if __name__ == "__main__":

    info_df = pd.read_csv(info_file)