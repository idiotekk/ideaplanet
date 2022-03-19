from random import uniform
import sys, os
import pprint
from PIL.Image import new
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
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

input_dir = Path("/Users/zche/data/0xgenerator/circles/inputs/")
output_dir = Path("/Users/zche/data/0xgenerator/circles/outputs/") 


def gen_iris():
    
    
    
    input_shape = (600, 600)
    input_h, input_w, *_ = input_shape
    xx, yy = frm.get_coords(input_shape)

    x_c, y_c = input_h // 2 + 0.5, input_w // 2 + 0.5 # dodge from integers
    x1, y1 = xx - x_c, yy - y_c # shift coordinate to center
    r = np.sqrt(x1**2 + y1**2)
    theta = np.arctan2(y1, x1)

    max_radius = max(input_w, input_h) // 2 * 0.95
    inner_radius_ratio = 0.3
    inner_radius = max_radius * inner_radius_ratio
    transition_width_ratio = 0.05
    transition_width = max_radius * transition_width_ratio
    lightness_ratio =  (
        ( 1 / ( 1 + np.exp( - ( r -  inner_radius ) / transition_width ) ))
    )

    #radius_ratios = np.sqrt(np.sort(np.random.uniform(0, 1, (n_rings)) ))
    #radius_ratios = np.sqrt(np.sort(np.linspace(0, 1, n_rings) ))
    radius_ratios = np.array([0.3, 0.4, 0.5, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.8, 0.9, 0.94])
    #thicknesses = np.random.randint(2, 4, (n_rings))
    thicknesses = np.array([10, 5, 5, 2, 2, 2, 2, 2, 2, 5, 3, 3])
    n_rings = len(radius_ratios)
    colors = np.stack([
        np.random.uniform(100, 250, (n_rings)),
        np.random.uniform(100, 250, (n_rings)),
        np.random.uniform(10, 150, (n_rings))
    ], axis=1)
        
    grids = np.random.randint(10, 30, (n_rings))
    angle_speeds = pi * 2 / grids * ((5 *
                                     np.random.uniform(0, 1., (n_rings))**3
                                     ).astype(int) + 1) * ( 
                                                           np.random.randint(0, 2, (n_rings))  * 2 - 1) # this decides the direction
    phases = np.random.uniform(0, pi * 2, (n_rings))
    phase_lambda = np.random.randint(1, 4, (n_rings))

    n_frames = 150
    output_arrs = []
    log.info("generating frames.....")
    for i in tqdm(range(n_frames)):
        t = i / n_frames
        new_arr = frm.uniform(size=input_shape, color=frm.BLACK)
        prev_radius = inner_radius
        for i, radius_ratio in enumerate(radius_ratios):
            color = colors[i,:]
            thetat = theta + angle_speeds[i] * t
            new_layer = frm.uniform(size=input_shape, color=color)
            layer_lightness_ratio = (np.cos(thetat * grids[i])**2 ) ** 0.1
            new_layer = layer_lightness_ratio.reshape((input_h, input_w, 1)) * new_layer
            total_lightness = ( np.sin( phase_lambda[i] * t * pi * 2 + phases[i]) + 1 ) / 2 * 0.7 + 0.3
            new_layer = new_layer * total_lightness
            radius = max_radius * radius_ratio
            """
            weight = np.maximum(
                    np.minimum(
                    r - np.max([radius - thicknesses[i], prev_radius + 1]),
                    radius - r
                    ), 0.0).reshape(
                        (input_h, input_w, 1)
                )
            weight = weight / np.max(weight)
            #print("-----------------")
            #print(np.max(weight), np.min(weight))
            #new_arr = weight * new_layer + new_arr * (1 - weight)
            new_arr = weight * new_layer + new_arr 
            """
            new_arr = frm.where(
                np.logical_and(
                    r - np.max([radius - thicknesses[i], prev_radius + 1]) > 0,
                    radius - r > 0
                ),
                new_layer,
                new_arr
            )
            prev_radius = radius
        new_arr = frm.where(r > max_radius, frm.BLACK, new_arr) 
        new_arr = new_arr * lightness_ratio.reshape((input_h, input_w, 1))

        #def circular_dist(x, y):
        #    return np.min([abs(x - y), abs(x - y - 1), abs(x - y +1)])
        #final_lightness = np.clip(circular_dist(t, 0) - 0.1, 0, 0.2) / 0.2
        #new_arr = new_arr * final_lightness
        output_arrs.append(new_arr)

    output_frames = []
    log.info("post processing frames.....")
    for i, arr in tqdm(enumerate(output_arrs)):
        frame = io.np_to_im(arr, "RGB")
        frame = frame.filter(ImageFilter.SMOOTH)
        output_frames.append(frame.quantize(kmeans=3))#dither=Image.NONE)

    output_file = Path(output_dir) / f"iris.gif"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=10.0,
    )

parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    #gen_iris(file_name=parser.parse_args().file_name, colored=False)
    gen_iris()
