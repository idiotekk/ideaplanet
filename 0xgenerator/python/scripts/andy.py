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
from PIL import ImageEnhance
from tqdm import tqdm
from argparse import ArgumentParser
log.basicConfig(level=log.DEBUG)

input_dir = Path("/Users/zche/cloud/data/0xgenerator/andy/inputs/")
output_dir = Path("/Users/zche/cloud/data/0xgenerator/andy/outputs/") 

from math import sqrt, cos, sin, pi

class HueRotator(object):

    def __init__(self):
        return

    def set_hue_rotation(self, theta):
        cos_a = cos(theta)
        sin_a = sin(theta)
        self.matrix = np.eye(3)
        self.matrix[0][0] = cos_a + (1.0 - cos_a) / 3.0
        self.matrix[0][1] = 1./3. * (1.0 - cos_a) - sqrt(1./3.) * sin_a
        self.matrix[0][2] = 1./3. * (1.0 - cos_a) + sqrt(1./3.) * sin_a
        self.matrix[1][0] = 1./3. * (1.0 - cos_a) + sqrt(1./3.) * sin_a
        self.matrix[1][1] = cos_a + 1./3.*(1.0 - cos_a)
        self.matrix[1][2] = 1./3. * (1.0 - cos_a) - sqrt(1./3.) * sin_a
        self.matrix[2][0] = 1./3. * (1.0 - cos_a) - sqrt(1./3.) * sin_a
        self.matrix[2][1] = 1./3. * (1.0 - cos_a) + sqrt(1./3.) * sin_a
        self.matrix[2][2] = cos_a + 1./3. * (1.0 - cos_a)
        log.info(f"set rotation angle as {theta}")

    #def apply(self, r, g, b):
    def apply(self, arr):
        _shape = arr.shape
        assert _shape[-1] == 3, "RGB color must be three numbers."
        res = np.tensordot(arr, self.matrix, axes=(len(_shape)-1, 1))
        res = np.clip(np.round(res), 0, 255)
        print(_shape, res.shape)
        return res

def rotate_rgb(arr, theta):
    """ Rotate an rgb image by theta.
    """
    hue_rotator = HueRotator()
    hue_rotator.set_hue_rotation(theta)
    new_arr = hue_rotator.apply(arr)
    return new_arr


def rotate_hsv(arr_rgb, theta):
    from matplotlib import colors
    arr_hsv = colors.rgb_to_hsv(arr_rgb)
    arr_hsv[:,:,2] = np.mod(arr_hsv[:,:,2]  + 360 * theta, 360)
    res = colors.hsv_to_rgb(arr_hsv)
    return res

    
def round_corner(arr, radius, bg_color):
    assert 0 <= radius < 0.5
    h, w = arr.shape[:2]
    x_coords = np.array([[i] * w for i in range(h)])
    y_coords = np.array([[j for j in range(w)]] * h)
    #log.info(x_coords)
    #log.info(y_coords)
    radius = min(h, w) * radius # convert radius to pixels
    inside_mask = np.array([[False] * w] * h)
    inside_mask = np.logical_or.reduce(
        (inside_mask, 
         np.logical_and(x_coords >= 0 + radius, x_coords <= h - radius - 1),
         np.logical_and(y_coords >= 0 + radius, y_coords <= w - radius - 1),
         np.logical_or.reduce(
                [
                    ((x_coords - xx) ** 2+(y_coords - yy) ** 2) <= radius ** 2 for xx, yy in (
                        (radius, radius),
                        (radius, w - radius),
                        (h - radius, radius),
                        (h - radius, w - radius)
                    )
                ]
         )))

    inside_mask = np.stack([inside_mask]*3, axis=2)
    bg = np.array([[bg_color]*w] * h)
    res = np.where(inside_mask,
                   arr,
                   bg
                   )
    return res


def gen_mutate_factors(n, fixed=10):
    """
    between zero and one.
    """

    mutate_factors = np.zeros((n,))

    while np.max([
        np.max(np.abs(np.diff(mutate_factors))),
        np.abs(mutate_factors[0] - mutate_factors[-1])
    ]) <= 4:
        log.info("retrying...")
        mutate_factors = np.random.choice(np.arange(1, n), n - 1, replace=False) 
        mutate_factors = np.random.choice(mutate_factors, len(mutate_factors), replace=False)
        mutate_factors = np.insert(mutate_factors, fixed, 0) * 1.0
    mutate_factors += ((np.random.rand() - 0.5) * 2) * 0.1 # add noise
    mutate_factors = mutate_factors / n  # normalizek
    mutate_factors[fixed] = 0.0 # middle is reset to 0

    log.info(mutate_factors)
    return mutate_factors

    
def gen_andy(file_name):

    input_frame = io.read_frame(input_dir / f"{file_name}", to_np=False) 
    #input_frame = input_frame.resize((300, 300))
    input_w, input_h = input_frame.size
    margin = 0.02
    margin0 = 0.01
    grid_w, grid_h = 4, 4
    output_size = (
        int(grid_h * input_h * (1 + 2*margin) + input_h * margin0 * 2), 
        int(grid_w * input_w * (1 + 2*margin) + input_w * margin0 * 2))

    # blank canvas
    bg_color = (255, 255, 255)
    output_im = Image.new("RGB", output_size, color=bg_color)

    idx_seq = [(i, j) for i in range(grid_w) for j in range(grid_h)]
    n_sub_ims = len(idx_seq)
    mutate_factors = gen_mutate_factors(n_sub_ims, fixed=10)

    xx = 2.0
    for idx, (i, j) in enumerate(idx_seq):
        log.info(f"sub image {(i, j)}")
        new_arr = np.array(input_frame)
        #new_arr = rotate_hsv(new_arr, rotate_thetas[idx])
        mutate_factor = mutate_factors[idx]
        new_arr = rotate_rgb(new_arr, mutate_factor * pi * 2)
        new_arr = round_corner(new_arr, 0.03, frm.WHITE)
        #for _ in range(3):
            #new_arr[:,:,_] = np.mod(new_arr[:,:,_] + np.random.rand() * 256 * mutate_factor, 256)
        new_im = io.np_to_im(new_arr, "RGB")
        if mutate_factor != 0:
            new_im = ImageEnhance.Sharpness(new_im).enhance(5 * mutate_factor)
            new_im = ImageEnhance.Contrast(new_im).enhance(xx)
            new_im = ImageEnhance.Brightness(new_im).enhance(0.8)
        #io.save_im(new_im, output_dir / "debug" / f"{i}_{j}.png")
        output_im.paste(new_im, (
            int(i * (input_h*(1+2*margin)) + input_h * margin + input_h * margin0), 
            int(j * (input_w*(1+2*margin)) + input_w * margin + input_w * margin0)
        ))

    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_4by4.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.save_im(output_im, output_file)
    output_im.show()


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_andy(file_name=parser.parse_args().file_name)
