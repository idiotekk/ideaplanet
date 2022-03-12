from asyncio import start_unix_server
import sys, os
import pprint
from tkinter import N
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
from typing import Tuple
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

input_dir = Path("/Users/zche/data/0xgenerator/quadtree/inputs/")
output_dir = Path("/Users/zche/data/0xgenerator/quadtree/outputs/") 

def gen_quadtree(file_name):

    input_frame = io.read_frame(input_dir / f"{file_name}", to_np=False)#, size=(2000, 2000)) 
    input_arr = np.array(input_frame)
    h, w = input_frame.size
    log.info(f"input size: {h}, {w}")
    reso = 100
    output_arr = np.clip( np.round(input_arr /reso) * reso, 0, 255)
    output_im = io.np_to_im(output_arr)
    io.save_im(output_im, output_dir / f"test_{file_name.split('.')[0]}.png")

    return

    #p = 1 for azuki
    p = 1.0
    min_size = 7
    root = Quad(base_arr=input_arr, x=0, y=0, h=h, w=w)
    root.split(p=p, min_size=min_size)
    print(root.n)

    n_frames = 100

    def render_quad_square(quad, canvas):
        canvas[(quad.x):(quad.x+quad.h), (quad.y):(quad.y+quad.w), :] = 50
        canvas[(quad.x+1):(quad.x+quad.h), (quad.y+1):(quad.y+quad.w), 0] = np.round(np.mean(quad.arr[:,:,0]))
        canvas[(quad.x+1):(quad.x+quad.h), (quad.y+1):(quad.y+quad.w), 1] = np.round(np.mean(quad.arr[:,:,1]))
        canvas[(quad.x+1):(quad.x+quad.h), (quad.y+1):(quad.y+quad.w), 2] = np.round(np.mean(quad.arr[:,:,2]))

    def render_quad(quad, canvas):
        default_value = 50
        x_c = quad.h / 2
        y_c = quad.w / 2
        xx, yy = frm.get_coords((quad.h, quad.w))
        replace = (np.abs((xx - x_c) / (quad.h / 2 ))**3 + np.abs((yy - y_c) / (quad.w / 2)) **3  < 0.98)
        canvas[(quad.x):(quad.x+quad.h), (quad.y):(quad.y+quad.w), 0] = np.where(replace, np.round(np.mean(quad.arr[:,:,0])), default_value)
        canvas[(quad.x):(quad.x+quad.h), (quad.y):(quad.y+quad.w), 1] = np.where(replace, np.round(np.mean(quad.arr[:,:,1])), default_value)
        canvas[(quad.x):(quad.x+quad.h), (quad.y):(quad.y+quad.w), 2] = np.where(replace, np.round(np.mean(quad.arr[:,:,2])), default_value)

    stack = [root]
    output_arrs = []

    canvas = frm.uniform(size=(h, w), color=frm.get_bg_color(input_arr))

    sample_idxs = np.unique(np.array([int((i / n_frames)**4 * root.n) for i in range(n_frames)] + [root.n]))

    count = 0
    output_arrs.append(canvas.copy())
    while stack:

        ps = np.array([_.size for _ in stack])
        quad = stack.pop(np.random.choice(range(len(stack)), p=ps / np.sum(ps)))
        if count in sample_idxs:
            output_arrs.append(canvas.copy())
            log.info(f"count={count}")
        count += 1
 
        if quad.can_split:
            for _ in quad.children:
                render_quad(_, canvas)
                stack.append(_)

       
    log.info(f"n frames = {len(output_arrs)}")
    output_frames=[]
    for i, arr in tqdm(enumerate(output_arrs)):
        frame = io.np_to_im(arr, "RGB")
        #output_frames[i] = frame #.quantize()#dither=Image.NONE)
        output_frames.append(frame.quantize(kmeans=3))#dither=Image.NONE)
    output_frames = ( [input_frame_bg] * (len(output_frames) // 4 ) 
                     + output_frames 
                     + [output_frames[-1]] * (len(output_frames) // 4) 
                     + [input_frame_bg] * (len(output_frames) // 4)
                     )
    log.info(f"n frames = {len(output_frames)}")

    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_heart.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"chunk_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=10.0,
    )


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_quadtree(file_name=parser.parse_args().file_name)
