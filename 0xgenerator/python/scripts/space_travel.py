import pandas as pd
from glob import glob
import sys, os
import pprint
import numpy as np
from PIL import ImageFilter
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
pprint.pprint(sys.path)
from py0xlab import *
from py0xlab import frame as frm
from py0xlab import io
from pathlib import Path
from PIL import ImageEnhance, ImageFilter
from tqdm import tqdm
from argparse import ArgumentParser
from math import pi
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

import numpy as np
import sys

class Star:

    cur_pos: np.array
    speed: np.array 
    init_pos: np.array # intial position
    respawn_z: float

    def __init__(self, *, init_pos, speed, respawn_z):
        self.init_pos = init_pos
        self.speed = speed
        self.respawn_z = respawn_z
    
    def cur_pos(self, t):
        pos = self.init_pos + t * self.speed
        pos[2] = pos[2] % self.respawn_z
        return pos

    def proj(self, t, proj_z):
        """ Reintroduce at respawn_z if goes negative.
        """
        x, y, z = self.cur_pos(t)
        return int(x / z * proj_z), int(y / z * proj_z)


output_dir = Path("/Users/zche/cloud/data/0xgenerator/space_travel/outputs/") 

def space_travel():

    n_stars = 10000
    proj_z = 10

    n_frames = 100
    n_cycles = 1
    total_time = 10 # seconds
    output_size = np.array([100]*2)
    max_visible_distance = 300
    box_depth = 300
    box_size = np.array([1000] * 2 + [box_depth]) # stars will live here, they will respawn after death, can't escape!!
    log.info({"box_size": box_size})
    base_speed = np.array([0, 0, -1]) * n_cycles * box_depth / total_time

    stars = []
    log.info("generating stars.....")
    for i in tqdm(range(n_stars)):
        init_pos = np.random.uniform(-1., 1., size=[3]) * box_size
        init_pos[2] = abs(init_pos[2]) + proj_z
        new_star = Star(
            init_pos = init_pos,
            speed = base_speed,
            respawn_z = box_depth,
        )
        stars.append(new_star)


    output_arrs = []
    log.info("generating frames.....")
    for i in tqdm(range(n_frames)):
        t = i / n_frames * total_time # seconds
        output_h, output_w = output_size
        xc, yc = output_h // 2, output_w // 2
        new_arr = frm.uniform(size=output_size, color=frm.BLACK)

        for star_idx, star in enumerate(stars):
            cur_pos = star.cur_pos(t)
            if cur_pos[2] > max_visible_distance:
                continue
                #stars.pop(star_idx) # stars that's gone behind the screen will be removed
            xp, yp = star.proj(t, proj_z) # projected location on canvas
            x_canvas = xc + xp
            y_canvas = yc + yp
            if not (0 <= x_canvas and x_canvas < output_h and 0 <= y_canvas and y_canvas < output_w):
                #log.info(f"star out of scope: {(xp, yp)}, {(x_canvas, y_canvas)}")
                pass
            else:
                dist_to_viewer = np.linalg.norm(cur_pos)
                lightness = max(0.0, -dist_to_viewer+max_visible_distance) / max_visible_distance
                #lightness = max(0.0, -cur_pos[2]+max_visible_distance) / max_visible_distance
                new_arr[x_canvas, y_canvas, :] = (frm.WHITE * lightness).astype(int)
        output_arrs.append(new_arr)

    output_frames = []
    log.info("post processing frames.....")
    for i, arr in tqdm(enumerate(output_arrs)):
        #if i / len(output_arrs) <= warmup_time / total_time: # not including warmup period
            #continue
        frame = io.np_to_im(arr, "RGB")
        output_frames.append(frame.quantize(kmeans=3))#dither=Image.NONE)
        enhancer = ImageEnhance.Contrast(frame)
        texture_im = enhancer.enhance(0.2)
        texture_im = texture_im.resize((500, 500), Image.NEAREST)

    output_file = output_dir / f"space_travel.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=5.0,
    )


parser = ArgumentParser()
#parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    space_travel()
