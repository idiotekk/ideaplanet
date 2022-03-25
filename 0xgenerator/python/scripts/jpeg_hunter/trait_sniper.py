import json, os, glob
from operator import index
from tqdm import tqdm
from argparse import ArgumentParser
import pathlib
import numpy as np
import pandas as pd
import requests
import logging as log
log.basicConfig(level=log.INFO)
# the 3 lines below make browser invisible
from py0xlab.test import Timer

trait_json_url = {
    "azuki": "https://ikzttp.mypinata.cloud/ipfs/QmQFkLSQysj94s5GvTHPyzTxrawwtjgiiYS2TBLgrvw8CW/",
}


def get_azuki_info(url):

    log.info(f"opening {url}")
    try_count = 0
    success = False
    while try_count <= 10 and not success:
        try:
            response = requests.get(url, timeout=0.5)
            log.info(f"{try_count}-th try succeeded")
            success = True
        except requests.exceptions.ReadTimeout:
            log.info(f"{try_count}-th try failed")
            try_count += 1
    asset_info = json.loads(response.text)
    row = pd.DataFrame({
        "name": asset_info["name"],
        "image": asset_info["image"],
    }, index=[num])
    for attr in asset_info["attributes"]:
        row[attr["trait_type"]] = attr["value"]

    return row


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--slug", default="azuki")
    parser.add_argument("--chunk-size", default=100)
    parser.add_argument("--start", "-s", type=int)
    parser.add_argument("--end", "-e", type=int)
    parser.add_argument("--combine", "-c", action="store_true")
    args = parser.parse_args()
    slug = args.slug
    output_dir = pathlib.Path("/Users/zche/data/0xgenerator/database/trait_info/") / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    root_url = trait_json_url[slug]

    chunk_size = args.chunk_size
    chunk_count = 0
    count = 0

    if args.combine:

        df = []
        for f in tqdm(glob.glob(str(output_dir / str(chunk_size)  / f"*.csv"))):
            df.append(pd.read_csv(f))
        df = pd.concat(df)
        df["num"] = df["name"].apply(lambda x: int(x.split("#")[-1]))
        df.to_csv(str(output_dir / "traits_incomplete.csv"))

        missing_nums = np.setdiff1d(np.arange(10000), df["num"].values)
        while len(missing_nums) > 0:
            log.info(f"missing {missing_nums}")
            df = [df]
            for num in missing_nums:
                log.info(f"getting missing item {num}")
                url = f"{root_url}/{num}"
                row = get_azuki_info(url)
                df.append(row)
            df = pd.concat(df)
            df["num"] = df["name"].apply(lambda x: int(x.split("#")[-1]))
            df = df.sort_values("num")
            missing_nums = np.setdiff1d(np.arange(10000), df["num"].values)
        df.to_csv(str(output_dir / "traits.csv"), index=False)
       
    else:
        assert args.start is not None and args.end is not None
        for chunk_idx in range(args.start, args.end):
            num_range = range(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size)
            log.info(f"downloading chunk {chunk_idx}: {num_range[0]} ot {num_range[-1]}")
            df = []
            for num in num_range:
                try:
                    with Timer(f"{slug} {num}"):
                        url = f"{root_url}/{num}"
                        log.info(f"opening {url}")
                        response = requests.get(url, timeout=0.5)
                        asset_info = json.loads(response.text)
                        row = pd.DataFrame({
                            "name": asset_info["name"],
                            "image": asset_info["image"],
                        }, index=[num])
                        for attr in asset_info["attributes"]:
                            row[attr["trait_type"]] = attr["value"]
                        df.append(row)
                except Exception as e:
                    log.warning(f"noooo!!! failed getting {slug} {num}")
                    log.warning(f"error: {e}")
            df = pd.concat(df)
            (output_dir / str(chunk_size)).mkdir(parents=True, exist_ok=True)
            df.to_csv(str(output_dir / str(chunk_size)  / f"{chunk_idx}.csv"))
            count = 0
            chunk_count += 1
            df = []