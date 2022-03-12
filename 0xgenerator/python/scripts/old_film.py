from asyncio import start_unix_server
from enum import Flag
from types import new_class
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
from py0xlab import io
from pathlib import Path
from PIL import ImageEnhance, ImageFilter
from tqdm import tqdm
from argparse import ArgumentParser
from math import pi
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


input_dir = Path("/Users/zche/cloud/data/0xgenerator/bg_switch/inputs/")
output_dir = Path("/Users/zche/cloud/data/0xgenerator/bg_switch/outputs/") 

def old_film(file_name):

    #base_arr = np.array(io.read_frame(input_dir / file_name, size=(300, 300), to_np=False).convert("L").convert("RGB"))
    base_frame = io.read_frame(input_dir / file_name, to_np=False).convert("L").convert("RGB")
    base_frame = base_frame.resize((600, 220))
    base_arr = np.array(base_frame)
    #base_arr = np.array(io.read_frame(input_dir / file_name, size=(300, 300), to_np=False))
    input_h, input_w = base_arr.shape[:2]
    #texture_im = io.read_frame(input_dir / "texture1.jpeg", size=segmented_image.shape[:2], to_np=False)
    #texture_ims = io.read_gif(input_dir / "oldfilm4.gif", mode="L", samples=50, to_np=False)
    texture_ims = io.read_gif(input_dir / "texture2.gif", mode="L", samples=50, to_np=False, square=False, size=None)
    texture_lightnesses = []
    for texture_im in texture_ims:
        #texture_im = io.read_frame(input_dir / "texture1.jpeg", to_np=False)
        _w, _h = texture_im.size
        ratio = min(_w / input_w, _h / input_h)
        _w, _h = int(ratio * input_w), int(ratio * input_h)
        texture_im = texture_im.crop((0, 0, _w, _h))
        texture_im = texture_im.resize((input_w, input_h))
        #enhancer = ImageEnhance.Brightness(texture_im)
        #texture_im = enhancer.enhance(3)
        #enhancer = ImageEnhance.Contrast(texture_im)
        #texture_im = enhancer.enhance(1.5)
        texture_im = texture_im.convert("L").convert("RGB")
        texture_lightness = np.array(texture_im) 
        texture_lightness = texture_lightness / np.max(np.array(texture_lightness) + 1)
        texture_lightnesses.append(texture_lightness)

    xx, yy = frm.get_coords((input_h, input_w))
    """
    dist_to_boundary = np.sqrt((
        (np.minimum(abs(xx), abs(input_h-xx)))**2 +
        (np.minimum(abs(yy), abs(input_w-yy)))**2 
    ))
    """
    dist_to_boundary = 1.0 / (
        1 / (np.minimum(abs(xx+1), abs(input_h-xx)))  +
        1 / (np.minimum(abs(yy+1), abs(input_w-yy))) 
    ) + np.random.rand(input_h, input_w) * 0.3
    edge_darkener = np.minimum(
        dist_to_boundary,
        10
    )
    edge_darkener = edge_darkener / np.max(edge_darkener)
    edge_darkener = np.stack([edge_darkener] * 3, axis=2)

    """
    for l in np.unique(labels):
        log.info(f"processing label {l} ... ")

        new_labels = [_ if _ == l else k for _ in labels]
        new_center = centers[l]
        new_mask = (np.array(new_labels) == l).reshape((input_h, input_w))

        # new texture
        new_texture_arr = (texture_lightness * new_center).astype(int)
        # random shift
        new_texture_arr = np.roll(new_texture_arr, (shift_x, shift_y), axis=(0, 1))

        # apply texture
        new_arr = frm.where(new_mask, new_texture_arr, new_arr)
        #output_file = Path(output_dir) / f"{file_name.split('.')[0]}_segmented/{l}.png"
        #output_file.parent.mkdir(parents=True, exist_ok=True)
        #io.save_im(new_im, output_file)

    new_im = io.np_to_im(new_arr, "RGB")
    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_textured.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.save_im(new_im, output_file)
    new_im.show()
    """
    
    n_frames = 30
    output_arrs = []

    texture_start_t = np.random.randint(n_frames) + n_frames
    log.info("generating frames.....")
    for t in tqdm(range(n_frames)):
        texture_lightness = texture_lightnesses[(t+texture_start_t) % len(texture_ims)]
        shift_x = 0# np.random.randint(0, input_h)
        shift_y = np.random.randint(0, input_w)
        new_texture_lightness = np.roll(texture_lightness, (shift_x, shift_y), axis=(0, 1))
        new_arr = base_arr * new_texture_lightness
        new_arr = new_arr * edge_darkener
        output_arrs.append(new_arr)

    output_frames = []
    for i, arr in tqdm(enumerate(output_arrs)):
        frame = io.np_to_im(arr, "RGB")
        output_frames.append(frame.quantize(kmeans=3))#dither=Image.NONE)
        enhancer = ImageEnhance.Contrast(frame)
        texture_im = enhancer.enhance(0.2)
        texture_im = texture_im.resize((500, 500), Image.NEAREST)

    output_file = Path("/Users/zche/cloud/data/0xgenerator/bg_switch/outputs/") / f"vintage_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=5.0,
    )



parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    #cv_cluster(file_name=parser.parse_args().file_name)
    #bg_switch(file_name=parser.parse_args().file_name)
    old_film(file_name=parser.parse_args().file_name)
