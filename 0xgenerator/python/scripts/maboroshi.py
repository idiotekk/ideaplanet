from asyncio import start_unix_server
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

input_dir = Path("/Users/zche/data/0xgenerator/maboroshi/inputs/")
output_dir = Path("/Users/zche/data/0xgenerator/maboroshi/outputs/") 

def bg_switch(file_name):

    #input_arr = io.read_frame(input_dir / file_name, to_np=True, size=(600, 600))
    #input_arr = io.read_frame(input_dir / file_name, to_np=True, size=(1121, 1121))
    input_arr = io.read_frame(input_dir / file_name, size=(600, 600), to_np=True)
    input_arr, input_labels = algo.clusterize(input_arr, k=15)
    input_bg_color = frm.get_bg_color(input_arr)
    is_bg_color = frm.is_same_color(input_arr, color=input_bg_color, tol=3)
    input_shape = input_arr.shape[:2]
    input_h, input_w = input_shape
    input_arr2 = io.read_frame(input_dir / "maboroshi.jpeg", to_np=True, size=(input_w, input_w))
    input_arr2, _ = algo.clusterize(input_arr2, k=2)
    input_arr2 = (256 - input_arr2) * 0.5
    xx, yy = frm.get_coords(input_shape)

    x_c1, y_c1 =  input_h  + 0.5, 0 + 0.5 # dodge from integers
    x1, y1 = xx - x_c1, yy - y_c1 # shift coordinate to center
    r1 = np.sqrt(x1**2 + y1**2).astype(int)
    x_c2, y_c2 = 0.5,  input_w + 0.5 # dodge from integers
    #x_c2, y_c2 = input_h // 2 + 0.5, input_w // 2 + 0.5 # dodge from integers
    x2, y2 = xx - x_c2, yy - y_c2 # shift coordinate to center
    r2 = np.sqrt(x2**2 + y2**2).astype(int)

    grid_size = 10
    n_frames = grid_size
    output_arrs = []

    log.info("generating frames.....")
    for t in tqdm(range(n_frames)):
        shift = int(grid_size * t  / n_frames)
        rt1 = r1 + shift
        rt2 = r2 + shift
        """
        mask = np.logical_or( 
                              ( rt1 % grid_size  == input_labels), 
                              ( rt2 % grid_size  == input_labels), 
            )
        """
        mask = ( rt1 % grid_size  == input_labels)
        output_arr_fg = frm.where(mask, input_arr2, input_arr)
        mask2 = np.logical_and( 
                    #np.logical_and(
                        #np.logical_and( (rt1 % grid_size )  < 5 , (rt1 % grid_size )  >= 0  ), 
                        np.logical_and( (rt2 % grid_size )  < 5 , (rt2 % grid_size )  >= 0  ), 
                    #),
                is_bg_color)
        output_arr_bg = frm.where(mask2, input_arr2, input_arr)
        output_arr = frm.where(is_bg_color, output_arr_bg, output_arr_fg)
        output_arrs.append(output_arr)

    output_frames = []
    for i, arr in tqdm(enumerate(output_arrs)):
        frame = io.np_to_im(arr, "RGB")
        output_frames.append(frame.quantize(kmeans=3))#dither=Image.NONE)
        enhancer = ImageEnhance.Contrast(frame)
        texture_im = enhancer.enhance(0.2)
        texture_im = texture_im.resize((500, 500), Image.NEAREST)

    output_file = output_dir / f"maboroshi_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=0.2,
    )


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    bg_switch(file_name=parser.parse_args().file_name)
