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
output_dir = Path("/Users/zche/data/0xgenerator/trait_hunter/") / slug
im_dir = root_dir / slug
info_file = "azuki_traits.csv"

gender_clothing_map = {
    "male": {
        "Maroon Yukata",
        "Blue Kimono",
        "Green Yukata",
        "Black Yukata",
        "White Layered Yukata",
        "White T-Shirt",
        "White Yukata",
        "Suit with Turtleneck",
        "Black Kimono",
        "Shinto Robe",
        "Kimono with Jacket",
        "Kimono with Straw Hat",
        "Kung Fu Shirt",
        "Brown Yukata",
        "White Blazer",
        "Red Kimono",
        "Black Blazer",
        "Yellow Jumpsuit",
        ".*Yukata",
        "Suikan",
        "Matsuri Happi",
        
    },
    "female": {
        "Turquoise Kimono",
        "Blue Floral Kimono",
        "Red Floral Kimono",
        "Turquoise Kimono with Bow",
        "Lavender Kimono with Bow",
        "Blue Kimono with Bow",
        "Light Armor",
        "Red Ninja Top",
        "Black Ninja Top",
        "Tank Top with Jacket",
        ".*Qipao.*",
        ".*Bikini.*",
        ".*Oversized Kimono",
    }
}

def gender_identifier():

    info_df = pd.read_csv(info_file)



def read_asset(num, **kw):
    return io.read_frame(root_dir / f"{num}.png", **kw)
    
if __name__ == "__main__":

    info_df = pd.read_csv(info_file)

    # trait count analysis
    trait_count = {}
    for col in info_df.columns:
        if info_df[col].nunique() < 200:
            trait_count[col] = info_df[col].nunique()
            log.info("--------------------"*3)
            log.info(f"category: {col}")
            log.info(info_df[col].nunique())
            log.info("\n" + pprint.pformat(info_df[col].value_counts().to_dict()))
            log.info("--------------------"*3)
    trait_count = pd.Series(trait_count)
    trait_count["total"] = trait_count.sum()
    print(trait_count)

    exit()
    nums = info_df[info_df["Clothing"] == "Vest"]["num"].to_list()
    arrs = []
    for num in tqdm(nums):
        arr = read_asset(num)
        arrs.append(arr)
    tensor = np.stack(arrs, axis=3)
    mean = np.mean(tensor, axis=3)
    im = io.np_to_im(mean.astype(int))
    io.save_im(im, output_file=output_dir / "Vest.mean.png")