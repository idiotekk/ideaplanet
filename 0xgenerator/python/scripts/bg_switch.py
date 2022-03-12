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
from py0xlab import io
from pathlib import Path
from PIL import ImageEnhance, ImageFilter
from tqdm import tqdm
from argparse import ArgumentParser
from math import pi
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

input_dir = Path("/Users/zche/cloud/data/0xgenerator/bg_switch/inputs/")
output_dir = Path("/Users/zche/cloud/data/0xgenerator/bg_switch/outputs/") 

def bg_switch(file_name):

    #input_arr = io.read_frame(input_dir / file_name, to_np=True, size=(600, 600))
    #input_arr = io.read_frame(input_dir / file_name, to_np=True, size=(1121, 1121))
    input_arr = io.read_frame(input_dir / file_name, to_np=True)
    bg_arrs = [io.read_frame(_) for _ in glob(str(input_dir / "azuki/*"))]
    np.random.shuffle(bg_arrs)
    bg_arrs.insert(4, input_arr)
    input_h, input_w = input_arr.shape[:2]
    
    margin0 = 0.01
    margin1 = 0.01
    grid_w, grid_h = 3, 3
    #grid_w, grid_h = 1, 1
    output_size = (
        int(grid_h * input_h * (1 + 2*margin1) + input_h * margin0 * 2), 
        int(grid_w * input_w * (1 + 2*margin1) + input_w * margin0 * 2))

    canvas_color = frm.get_rgb("white")
    output_im = Image.new("RGB", output_size, color=canvas_color)

    xx, yy = frm.get_coords(input_arr.shape)
    input_bg_color = frm.get_bg_color(input_arr)
    is_bg_color = frm.is_same_color(input_arr, color=input_bg_color, tol=3)
    is_black = (
        np.logical_or.reduce((
            frm.is_same_color(input_arr, color=frm.BLACK, tol=30),
            frm.is_same_color(input_arr, color=frm.WHITE, tol=150),
            input_arr[:,:,2] > input_arr[:,:,1], # blue-ish
            )
        )
    )

    # flag2: has yellow neighbor
    yellow_neighbor_count = np.zeros(is_bg_color.shape)
    rr = 2
    count = 0
    for i in range(-rr, rr+1):
        for j in range(-rr, rr+1):
            yellow_neighbor_count += np.roll(is_bg_color, (i, j), (0, 1))
            count += 1
    all_neighbors_are_yellow  = (yellow_neighbor_count >= count)
    #is_isolated_yellow  = (yellow_neighbor_count <= count*0.2)
    visited = all_neighbors_are_yellow
    print(np.sum(visited) - np.sum(is_bg_color))

    flag = is_bg_color.copy()
    is_boundary = np.logical_and(False, flag) # all false

    def spread(i, j, depth, visited):
        if visited[i, j] == True:
            return
        if depth <= 0: 
            return
        for di in range(-1, 2):
            for dj in range(-1, 2):
                i_new = i + di
                j_new = j + dj
                if i_new < 0 or i_new >= input_h or j_new < 0 or j_new >= input_w:
                    continue
                if is_black[i_new, j_new] == False or np.logical_and(
                    input_arr[i_new,j_new,2] < input_arr[i_new,j_new,1] < input_arr[i_new,j_new,0],
                    j > input_w * 0.6
                ): # yellow-ish
                    flag[i_new, j_new] = True
                    if is_bg_color[i_new, j_new] == False:
                        is_boundary[i_new, j_new] = True
                    spread(i_new, j_new, depth - 1, visited)
        visited[i, j] = True

    for i in tqdm(range(input_h)):
        for j in range(input_w):
            if is_bg_color[i, j] == True and visited[i, j] == False:
                if i <= input_h // 2:
                    spread(i, j, 50, visited)
                else:
                    spread(i, j, 2, visited)

    idx_seq = [(i, j) for i in range(grid_w) for j in range(grid_h)]
    for idx, (i, j) in tqdm(enumerate(idx_seq)):

        bg_arr = bg_arrs[idx]
        bg_color = frm.get_bg_color(bg_arr)
        new_arr = frm.where(flag, bg_color, input_arr)
        blurred_arr = np.array(io.np_to_im(new_arr).filter(ImageFilter.GaussianBlur))
        new_arr = frm.where(is_boundary, blurred_arr, new_arr)
        #new_arr = frm.where(flag, bg_color, frm.WHITE) # test
        #new_arr = frm.where(is_black, frm.BLACK, new_arr) # test
        #test_row = int(input_h * 0.35)
        #pd.DataFrame(input_arr[input_h // 2,:,:], columns=["r", "g", "b"]).to_csv(str(output_dir/"inputslice.csv"))
        #pd.DataFrame(new_arr[test_row,:,:], columns=["r", "g", "b"]).to_csv(str(output_dir/"slice.csv"))
        #new_arr[test_row,:,:] = canvas_color
        pd.DataFrame(flag.astype(float)).to_csv(str(output_dir/"flag.csv"))
        #new_arr = frm.round_corner(new_arr, 0.02, bg_color=canvas_color)
        new_im = io.np_to_im(new_arr, "RGB")
        output_im.paste(new_im, (
            int(i * (input_h*(1+2*margin1)) + input_h * margin1 + input_h * margin0), 
            int(j * (input_w*(1+2*margin1)) + input_w * margin1 + input_w * margin0)
        ))

    #output_im = output_im.filter(ImageFilter.SMOOTH_MORE)
    output_im = output_im.filter(ImageFilter.SMOOTH)
    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_all.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.save_im(output_im, output_file)
    output_im.show()


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    bg_switch(file_name=parser.parse_args().file_name)
