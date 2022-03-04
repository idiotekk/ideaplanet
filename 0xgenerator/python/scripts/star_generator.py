from asyncio import start_unix_server
import sys, os
import pprint
from tokenize import blank_re
import numpy as np
from PIL import ImageFilter
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
pprint.pprint(sys.path)
from py0xlab import *
from py0xlab.frame import where, is_same_color, get_bg_color
from py0xlab import frame as frm
from py0xlab import io
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from math import pi
log.basicConfig(level=log.DEBUG)

class Star:

    arr: np.array
    center: tuple
    angle_speed: float
    curr_angle: float
    radius: float
    size: float
    tail_width: float

    def __init__(self):
        pass

    def loc(self, t):
        cx, cy = self.center
        return (
            cx + self.radius * np.cos(self.curr_angle),
            cy + self.radius * np.sin(self.curr_angle),
        )

def replace_non_background_by(arr1, arr2, bg_color1=None, bg_color2=None):
    """ Replace the backgroud in arr1 by non-background in arr2.
    """
    assert arr1.shape == arr2.shape, f"inputs shape mismatch: {arr1.shape} vs {arr2.shape}"
    if bg_color1 is None:
        bg_color1   = arr1[5, 5, :]
    if bg_color2 is None:
        bg_color2   = arr2[3, 3, :]
    is_bg1      = is_same_color(arr1, bg_color1)
    is_bg2      = is_same_color(arr2, bg_color2)
    #replace     = np.logical_and(is_bg1, np.logical_not(is_bg2))
    replace     = np.logical_not(is_bg1)
    new_arr     = where(replace, arr2, arr1)
    return new_arr

def replace_background_by(arr1, arr2, bg_color1=None, bg_color2=None):
    """ Replace the backgroud in arr1 by non-background in arr2.
    """
    assert arr1.shape == arr2.shape, f"inputs shape mismatch: {arr1.shape} vs {arr2.shape}"
    if bg_color1 is None:
        bg_color1   = arr1[5, 5, :]
    if bg_color2 is None:
        bg_color2   = arr2[3, 3, :]
    is_bg1      = is_same_color(arr1, bg_color1)
    is_bg2      = is_same_color(arr2, bg_color2)
    #replace     = np.logical_and(is_bg1, np.logical_not(is_bg2))
    replace     = is_bg1
    new_arr     = where(replace, arr2, arr1)
    return new_arr
    

def gen_star_gif(file_name):

    #input_file = Path(os.path.expandvars("~/cloud/data/0xgenerator/inputs")) / "sakura.png"
    input_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/inputs/")
    output_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") 
    output_size = (600, 600)
    #output_size = (1200, 1200)
    w, h = output_size
    fg_arr = io.read_frame(input_dir / f"{file_name}", size=output_size)

    n_stars = 100
    center = (w * -0.2, h * -0.3)

    base_speed = 0.2
    stars = []
    for i in tqdm(range(n_stars)):

        star = Star()
        star.center = center
        star.angle_speed = 0.1 + np.abs(pi * 2 * np.random.randn()) * base_speed # how many circles it travels in lifetime
        star.curr_angle = np.random.randn() * 2 * pi / 4
        star.size = 3
        star.color = frm.YELLOW
        star.radius = i / n_stars * h * 2
        star.tail_width = 2
        stars.append(star)

    n_frames = 100
    output_frames = []

    #decay_factor = 0.9
    #new_bg = frm.get_rgb("black", size=output_size)
    new_bg = frm.uniform(color=(0, 0, 0), size=output_size)
    xx, yy = frm.get_coords(size=output_size)
    for t in tqdm(range(n_frames)):
        new_snapshot = frm.uniform(color=(0, 0, 0), size=output_size)
        for s in stars:
            star_x, star_y = s.loc(t / n_frames)
            if star_x < 0:
                s.curr_angle = 0
                star_x, star_y = s.loc(t / n_frames)
            s.curr_angle += s.angle_speed * 1 / n_frames
            new_snapshot = frm.where(
                np.logical_or(
                    np.sqrt((xx - star_x)**2 + (yy - star_y)**2) <= s.size,
                    np.logical_and(
                        np.sqrt((xx - center[0])**2 + (yy - center[1])**2) < s.radius + s.tail_width / 2,
                        np.sqrt((xx - center[0])**2 + (yy - center[1])**2) > s.radius - s.tail_width / 2,
                    )
                ),
                np.array(s.color),
                new_snapshot
            )
        #new_bg = new_snapshot * ( 1 - decay_factor) + new_bg * decay_factor
        new_bg = new_snapshot
        arr = replace_non_background_by(
            fg_arr, 
            new_bg,
            bg_color1=get_bg_color(fg_arr),
            bg_color2=frm.YELLOW,
        )
        output_frames.append(arr)
    
    for i, arr in tqdm(enumerate(output_frames)):
        frame = io.np_to_im(arr, "RGB")
        output_frames[i] = frame.quantize(kmeans=3)
        #output_frames[i] = frame

    output_file = output_dir / f"star_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=7.0,
    )
    


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_star_gif(file_name=parser.parse_args().file_name)
