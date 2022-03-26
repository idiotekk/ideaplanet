from asyncio import start_unix_server
from distutils.log import info
import re
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
log.basicConfig(level=log.WARNING)
np.set_printoptions(edgeitems=8, linewidth=100000)

#parser = ArgumentParser()
#parser.add_argument("file_name", type=str)

slug = "azuki"
root_dir = Path("/Users/zche/data/0xgenerator/database/")
output_dir = Path("/Users/zche/data/0xgenerator/trait_hunter/") / slug
info_df = pd.read_csv(str(output_dir / "gender_likelihood.csv"))

def read_asset(num, **kw):
    return io.read_frame(root_dir / slug / f"{num}.png", **kw)
    
if __name__ == "__main__":

    arrs = []

    n_samples = 1000
    random_sample = {
        "boy": np.random.choice(info_df[info_df["gender_learned"] > 0]["num"].values, n_samples),
        "girl": np.random.choice(info_df[info_df["gender_learned"] < 0]["num"].values, n_samples),
        "pixie": np.random.choice(info_df[info_df["Hair"].fillna("").str.contains("ixie")]["num"].values, n_samples),
    }
    #for group_name in ["boy", "girl"]:
    for group_name in ["pixie"]:
        nums = random_sample[group_name]
        arr = None
        count = 0
        for num in tqdm(nums):
            try:
                new_arr = read_asset(num) * 1.0
            except:
                log.info(f"failed to read {num}")
                continue
            if arr is None:
                arr = new_arr
            else:
                arr = new_arr + arr
            count += 1
        mean = arr / count
        im = io.np_to_im(mean.astype(int))
        io.save_im(im, output_file=output_dir / f"{group_name}.png")