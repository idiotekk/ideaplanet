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


def cv_cluster(file_name):

    # read the image
    image = cv2.imread(str(input_dir / file_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_h, input_w = image.shape[:2]
    log.info({"height": input_h, "width": input_w})
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 10
    compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    centers_and_bg = np.row_stack([centers, np.array([[255, 255, 255]])])
    print(centers.shape, centers_and_bg.shape)
    # convert all pixels to the color of the centroids
    #centers = centers[np.random.permutation(centers.shape[0]), :]
    segmented_image = centers[labels]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    """
    output_im = io.np_to_im(segmented_image, "RGB")
    #output_im = output_im.filter(ImageFilter.SMOOTH_MORE)
    output_im = output_im.filter(ImageFilter.SMOOTH)
    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_segmented.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.save_im(output_im, output_file)
    output_im.show()

    """

    new_arr = segmented_image.copy()
    return


def add_texture(file_name):

    base_arr = np.array(io.read_frame(input_dir / file_name, size=(600, 600), to_np=False).convert("L").convert("RGB"))
    input_h, input_w = base_arr.shape[:2]
    #texture_im = io.read_frame(input_dir / "texture1.jpeg", size=segmented_image.shape[:2], to_np=False)
    texture_im = io.read_frame(input_dir / "texture1.jpeg", to_np=False)
    _w, _h = texture_im.size
    texture_im = texture_im.crop((0, 0, min(_w, _h), min(_w, _h)))
    texture_im = texture_im.resize((input_h, input_w))
    enhancer = ImageEnhance.Contrast(texture_im)
    texture_im = enhancer.enhance(1.5)
    texture_im = texture_im.convert("L").convert("RGB")
    texture_lightness = np.array(texture_im) 
    texture_lightness = texture_lightness / np.max(np.array(texture_lightness) + 1)

    shift_x = np.random.randint(0, input_h)
    shift_y = np.random.randint(0, input_w)

    

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
    
    n_pieces = 30
    output_arrs = []

    for t in tqdm(range(n_pieces)):
        shift_x = 0# np.random.randint(0, input_h)
        shift_y = np.random.randint(0, input_w)
        new_texture_lightness = np.roll(texture_lightness, (shift_x, shift_y), axis=(0, 1))
        new_arr = base_arr * new_texture_lightness
        output_arrs.append(new_arr)

    output_frames = []
    for i, arr in tqdm(enumerate(output_arrs)):
        frame = io.np_to_im(arr, "RGB")
        output_frames.append(frame.quantize(kmeans=3))#dither=Image.NONE)

    output_file = Path("/Users/zche/cloud/data/0xgenerator/bg_switch/outputs/") / f"vintage_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=7.0,
    )


def bg_switch(file_name):

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


def old_film(file_name):

    #base_arr = io.read_frame(input_dir / file_name, size=(600, 600))
    base_arr = np.array(io.read_frame(input_dir / file_name, size=(300, 300), to_np=False).convert("L").convert("RGB"))
    #base_arr = np.array(io.read_frame(input_dir / file_name, size=(300, 300), to_np=False))
    input_h, input_w = base_arr.shape[:2]
    #texture_im = io.read_frame(input_dir / "texture1.jpeg", size=segmented_image.shape[:2], to_np=False)
    texture_ims = io.read_gif(input_dir / "texture2.gif", mode="L", size=(input_h, input_w), samples=50, to_np=False)
    texture_lightnesses = []
    for texture_im in texture_ims:
        #texture_im = io.read_frame(input_dir / "texture1.jpeg", to_np=False)
        _w, _h = texture_im.size
        texture_im = texture_im.crop((0, 0, min(_w, _h), min(_w, _h)))
        texture_im = texture_im.resize((input_h, input_w))
        enhancer = ImageEnhance.Contrast(texture_im)
        texture_im = enhancer.enhance(1.5)
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
    print(edge_darkener[0,:])
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
        texture_lightness = texture_lightnesses[(t+texture_start_t) % n_frames]
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
