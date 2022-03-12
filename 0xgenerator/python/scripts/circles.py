from asyncio import start_unix_server
from dataclasses import replace
from random import uniform
import sys, os
import pprint
import numpy as np
from PIL import ImageFilter
from pandas import read_parquet
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
from py0xlab import *
from py0xlab.frame import where, is_same_color, get_bg_color
from py0xlab import frame as frm
from py0xlab import io
from pathlib import Path
from PIL import ImageEnhance, ImageFilter
from tqdm import tqdm
from argparse import ArgumentParser
from math import pi
from scipy.ndimage import gaussian_filter, uniform_filter
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

input_dir = Path("/Users/zche/data/0xgenerator/circles/inputs/")
output_dir = Path("/Users/zche/data/0xgenerator/circles/outputs/") 


def _3(x):
    return np.stack([x]*3, axis=2)

    
def gen_circles(file_name, colored):

    #input_frame = io.read_frame(input_dir / f"{file_name}", to_np=False)
    input_arr = io.read_frame(input_dir / f"{file_name}", to_np=True)
    log.info(f"image size : {input_arr.shape}")
    input_arr = frm.where(frm.is_same_color(input_arr, frm.get_bg_color(input_arr)),
                          np.array([0]*3),
                          input_arr)
    input_frame = io.np_to_im(input_arr)
    bw_frame = input_frame.convert("L") # bw = black and white
    bw_arr = np.array(bw_frame)
    log.info(f"image size : {input_frame.size}")

    input_h, input_w, *_ = bw_arr.shape
    xx, yy = frm.get_coords(bw_arr.shape)
    
    #signature_arr = io.read_frame(input_dir / f"signature.jpg", to_np=False).convert("L")
    signature_arr = io.read_frame(input_dir / f"signature.png", to_np=False)
    #signature_arr = signature_arr.quantize(kmeans=3).convert("RGB")
    signature_arr = np.array(signature_arr)
    signature_arr = signature_arr[-input_h:,-input_w:]
    #signature_arr = _3(signature_arr)
    signature_arr = np.where(signature_arr > 240, 255, 0)
    signature_arr = np.roll(signature_arr, (30, 25), axis=(0, 1))

    max_radius = max(input_w, input_h) // 2
    min_radius = max_radius * 0.1

    outer_ring1_radius = 0.99 * max_radius
    outer_ring0_radius = 0.97 * max_radius
    inner_ring0_radius = 1/3.6 * max_radius
    inner_ring1_radius = inner_ring0_radius #+ 4
    #hole_radius = 9/32/12 * max_radius
    hole_radius = 9/32/12 * max_radius / 2
    print(hole_radius, max_radius)

    r_step = 0.011 * max_radius
    x_c, y_c = input_h // 2 + 0.5, input_w // 2 + 0.5 # dodge from integers
    x1, y1 = xx - x_c, yy - y_c # shift coordinate to center
    r = np.sqrt(x1**2 + y1**2)
    theta = np.where(
        x1 > 0,
        np.arcsin(y1 / r),
        np.arcsin(-y1 / r) + pi,
    )
    #r_ref = (np.round(r / r_step) + (theta + pi / 2) / (2*pi)) * r_step
    adj_r = (theta + pi / 2) / (2*pi) * r_step
    r_ref = (np.round(r / r_step) ) * r_step
    x_ref = np.clip( ( x_c + r_ref * np.cos(theta) ).astype(int), 0, input_h - 1)
    y_ref = np.clip( ( y_c + r_ref * np.sin(theta) ).astype(int), 0, input_w - 1)
    color_ref = bw_arr.copy()
    for i in range(input_h):
        for j in range(input_w):
            color_ref[i, j] = bw_arr[x_ref[i,j], y_ref[i,j]]
    thickness_pct = (1 - (color_ref / 255 )) * 0.7 + 0.3 
    thickness_pct = gaussian_filter(thickness_pct, sigma=2)
    thickness = thickness_pct * r_step / 2
    replace = np.logical_and.reduce((
        r < max_radius,
        np.abs(r - r_ref) < thickness,
    ))
    replace = np.stack([replace] * 3, axis=2)

    margin0 = 0.01
    margin1 = 0.01
    grid_w, grid_h = 1, 1
    output_size = (
        int(grid_h * input_h * (1 + 2*margin1) + input_h * margin0 * 2), 
        int(grid_w * input_w * (1 + 2*margin1) + input_w * margin0 * 2))
    bg_color = frm.get_rgb("white")
    print(bg_color)
    output_im = Image.new("RGB", output_size, color=bg_color)

    # distance to nearest center
    # distance = np.sqrt((xx - xx_c) ** 2 + (yy - yy_c) ** 2)
    idx_seq = [(i, j) for i in range(grid_w) for j in range(grid_h)]

    colors = (
        #("yellow", (0, 187, 183)),
        ("yellow", "yellow"),
        ("white", "pink"),
        ("white", "black"),
        ("black", (255, 223, 100))
    )

    for idx, (i, j) in enumerate(idx_seq):

        bg_color, dot_color = colors[idx]
        if isinstance(bg_color, str):
            bg_color = frm.get_rgb(bg_color)
        if isinstance(dot_color, str):
            dot_color = frm.get_rgb(dot_color)        
        if colored:
            new_arr = np.where(replace, frm.BLACK, input_arr)
        else:
            new_arr = np.where(replace, frm.BLACK, frm.WHITE)
        #new_arr_color = np.where(replace, dot_color, frm.WHITE)
        new_arr_color = new_arr.copy()
        new_arr = np.where(_3(r > outer_ring0_radius), frm.BLACK, new_arr) # hole is white
        new_arr = np.where(_3(r > outer_ring1_radius), dot_color, new_arr) # hole is white
        new_arr = np.where(_3(r < inner_ring1_radius), dot_color, new_arr) # hole is white
        new_arr = np.where(_3(r < inner_ring0_radius), new_arr_color, new_arr) # hole is white
        new_arr = np.where(_3(r < hole_radius), frm.WHITE, new_arr) # hole is white
        
        new_arr = frm.where(
            frm.is_same_color(signature_arr, frm.get_bg_color(signature_arr)),
            new_arr,
            signature_arr
        )

        new_arr = gaussian_filter(new_arr, sigma=0.5)
        new_arr = frm.round_corner(new_arr, 0.01, bg_color=frm.WHITE)
        new_im = io.np_to_im(new_arr, "RGB")

        output_im.paste(new_im, (
            int(i * (input_h*(1+2*margin1)) + input_h * margin1 + input_h * margin0), 
            int(j * (input_w*(1+2*margin1)) + input_w * margin1 + input_w * margin0)
        ))

    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_circles_colored_{colored}.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.save_im(output_im, output_file)
    output_im.show()


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_circles(file_name=parser.parse_args().file_name, colored=False)
