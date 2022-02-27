from asyncio import start_unix_server
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
log.basicConfig(level=log.DEBUG)
np.set_printoptions(edgeitems=8, linewidth=100000)

input_dir = Path("/Users/zche/cloud/data/0xgenerator/polkadot/inputs/")
output_dir = Path("/Users/zche/cloud/data/0xgenerator/polkadot/outputs/") 

def get_distance(xx, yy, xx_c, yy_c, method="l2"):

    if method == "l2":
        distance = np.sqrt((xx - xx_c) ** 2 + (yy - yy_c) ** 2)
        return distance
    if method == "l1":
        distance = np.abs(xx - xx_c) + np.abs(yy - yy_c)
        return distance
    else:
        raise ValueError(f"unsupported : {method}")
    
def gen_polkadot(file_name):

    input_frame = io.read_frame(input_dir / f"{file_name}", to_np=False) 
    #input_frame = input_frame.filter(ImageFilter.SMOOTH)
    input_frame = input_frame.quantize(kmeans=20)
    bw_frame = input_frame.convert("L")
    bw_arr = np.array(bw_frame)
    log.info(f"image size : {input_frame.size}")
    grid_size = 15

    ## test
    #bw_arr = np.random.randint(0, 255, size=(10, 10))
    #grid_size = 5

    input_w, input_h, *_ = bw_arr.shape
    xx, yy = frm.get_coords(bw_arr.shape)
    log.info(f"xx : {xx}, yy: {yy}")
    margin0 = 0.01
    margin1 = 0.01
    grid_w, grid_h = 2, 2
    output_size = (
        int(grid_h * input_h * (1 + 2*margin1) + input_h * margin0 * 2), 
        int(grid_w * input_w * (1 + 2*margin1) + input_w * margin0 * 2))
    bg_color = frm.get_rgb("white")
    print(bg_color)
    output_im = Image.new("RGB", output_size, color=bg_color)

    # nearest center coords, clpped by bounds (need better treatment)
    xx_c = np.minimum(( (xx // grid_size) * grid_size + grid_size // 2).astype(int), input_w - 1 )
    yy_c = np.minimum(( (yy // grid_size) * grid_size + grid_size // 2).astype(int), input_h - 1 )

    #log.info(f"xx_c : {xx_c}, yy_c: {yy_c}")
    #log.info(f"xx_c : {xx_c[:,0]}, yy_c: {yy_c[0,:]}")
    color_c = bw_arr[xx_c[:,0], :][:, yy_c[0,:]]
   
    # distance to nearest center
    # distance = np.sqrt((xx - xx_c) ** 2 + (yy - yy_c) ** 2)
    idx_seq = [(i, j) for i in range(grid_w) for j in range(grid_h)]

    colors = (
        ("yellow", (0, 187, 183)),
        ("white", "pink"),
        ("white", "black"),
        ("black", (255, 223, 0))
    )
    distance_methods = (
        "l2",
        "l2",
        "l1",
        "l1"
    )
    radius_scale = (
        1.0,
        0.8,
        1.0,
        1.0
    )
    radius_floor = (
        0.5,
        0.0,
        0.0,
        0.3
    )
    for idx, (i, j) in enumerate(idx_seq):
        #output_arr = np.where(rgb_mask, frm.get_rgb("pink"), frm.get_rgb("white"))
        #output_arr = np.where(rgb_mask, frm.get_rgb("white"), frm.get_rgb("yellow"))
        radius = grid_size // 2 * np.maximum(
            (
            ( 1 - color_c / 256 ) * 0.8 + 0.2
        ) * radius_scale[idx], radius_floor[idx]) # rescale radius, then floor
        distance = get_distance(xx, yy, xx_c, yy_c, method=distance_methods[idx])
        #log.info(f"distance: {distance}")
        mask = ( distance < radius )
        rgb_mask = np.stack([mask] * 3, axis=2)

        bg_color, dot_color = colors[idx]
        if isinstance(bg_color, str):
            bg_color = frm.get_rgb(bg_color)
        if isinstance(dot_color, str):
            dot_color = frm.get_rgb(dot_color)

        log.info((bg_color, dot_color, i, j))

        new_arr = np.where(rgb_mask, dot_color, bg_color)
        new_arr = frm.round_corner(new_arr, 0.01, bg_color=frm.WHITE)
        new_im = io.np_to_im(new_arr, "RGB")

        output_im.paste(new_im, (
            int(i * (input_h*(1+2*margin1)) + input_h * margin1 + input_h * margin0), 
            int(j * (input_w*(1+2*margin1)) + input_w * margin1 + input_w * margin0)
        ))

    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_bw.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.save_im(output_im, output_file)
    output_im.show()


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_polkadot(file_name=parser.parse_args().file_name)
