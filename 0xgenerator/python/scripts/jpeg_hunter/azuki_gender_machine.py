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
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

#parser = ArgumentParser()
#parser.add_argument("file_name", type=str)

slug = "azuki"
root_dir = Path("/Users/zche/data/0xgenerator/database/")
output_dir = Path("/Users/zche/data/0xgenerator/trait_hunter/") / slug
im_dir = root_dir / slug
info_file = root_dir / "trait_info" / slug / "traits.csv"
info_df = pd.read_csv(info_file)

gender_clothing_regex_map = {
    "m": [
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
    ],
    "f": [
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
    ],
    "n": [
        
    ]
}

def gender_identifier():
    
    trait_gender_map = {}
    col = "Clothing"
    init_map = {
        "m": [],
        "f": [],
        "n": []
    }
    trait_gender_map[col] = init_map
    unique_values = info_df[col].dropna().unique()
    info_df["gender_ground_truth"] = 0
    info_df["gender_score"] = 0
    for value in unique_values:
        for m_value in gender_clothing_regex_map["m"]:
            if re.match(m_value, value):
                log.info(f"m trait: {value}")
                trait_gender_map[col]["m"].append(value)
                info_df.loc[info_df[col] == value, "gender_ground_truth"] = 1
        for f_value in gender_clothing_regex_map["f"]:
            if re.match(f_value, value):
                log.info(f"f trait: {value}")
                trait_gender_map[col]["f"].append(value)
                info_df.loc[info_df[col] == value, "gender_ground_truth"] = -1


    def bootstrap(old_col, new_col):
        truth = info_df[old_col]
        log.info(f"{truth.value_counts()}")
        is_m = truth > 0
        is_f = truth < 0
        if new_col == "gender_ground_truth":
            raise ValueError("cannot override ground truth")
        gender_score = info_df[old_col].copy() * 0
        for col in info_df.columns:
            if col.startswith("gender"):
                continue
            if info_df[col].nunique() > 200:
                log.info(f"{col} seems not a trait, skipping")
                continue
            trait_gender_map[col] = init_map
            log.info(f"checking trait category {col}")
            unique_values = info_df[col].unique()
            for value in unique_values:
                mask = (info_df[col] == value)
                if not np.any(mask & is_f):
                    log.info(f"likely m trait: {value}")
                    gender_score[mask] += np.sum(mask)
                elif not np.any(mask & is_m):
                    log.info(f"likely f trait: {value}")
                    gender_score[mask] -= np.sum(mask)
                else:
                    log.info(f"n trait: {value}")
                    trait_gender_map[col]["n"].append(value)
        gender_score_removed_known = gender_score * (truth == 0)
        gender_score_rank = gender_score_removed_known.rank() / len(gender_score_removed_known)
        unknown_but_is_obvious = (gender_score_rank < 0.05) | (gender_score_rank > 0.95)
        inferred = truth.copy()
        inferred[unknown_but_is_obvious] = np.where(gender_score_removed_known[unknown_but_is_obvious] > 0, 1, -1)
        summary = pd.DataFrame({
            old_col: truth,
            "gender_socre_rm_known": gender_score_removed_known,
            "gender_socre": gender_score,
            "unknown_but_is_obvious": unknown_but_is_obvious,
            "gender_score_rank": gender_score_rank,
            "inferred": inferred
        })
        print( summary, summary.describe()); 
        log.info({
            "old learned: ": np.sum(abs(truth)) ,
            "newly learned: ": np.sum(abs(inferred)) - np.sum(abs(truth)) ,
            "total: ": np.sum(abs(inferred)) ,
        })
        info_df[new_col] = inferred

    info_df["gender_learned_0"] = info_df["gender_ground_truth"]
    for i in range(11):
        bootstrap(f"gender_learned_{i}", f"gender_learned_{i+1}")
    log.info(f"output to" + str(output_dir / "gender_likelihood.csv"))
    info_df.to_csv(output_dir / "gender_likelihood.csv")


    
if __name__ == "__main__":

    gender_identifier()
