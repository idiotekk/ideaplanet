from mimetypes import init
import pandas as pd
from glob import glob
import sys, os
import pprint
import numpy as np
from PIL import ImageFilter, ImageDraw
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
from scipy.spatial.transform import Rotation

import numpy as np
import sys

class Point3D(np.array):
    
    def __init__(self, coords):
        assert len(coords) == 3, "coords must have 3 elemeents"
        self._coords = coords

    def rotate(self, center, umat):
        return Point3D(np.matmul(umat, self._coords - center) + center)

    def shift(self, v):
        if isinstance(v, Point3D):
            v = v.coords
        return Point3D(self._coords + v)

    def proj(self, z):
        """ Project this point to a plane defined by z (parallel to x-y plane)
        """
        x_, y_, z_ = self._coords
        return Point2D(int(x_ / z_ * z), int(y_ / z_ * z))

    def coords(self):
        return self._coords


class Point2D:

    def __init__(self, coords):
        assert len(coords) == 2, "coords must have 2 elemeents"
        self._coords = coords

    def coords(self):
        return self._coords


class Line2D:

    #cur_pos: np.array
    start: Point3D
    end: Point3D

    def __init__(self, *, 
                 start: Point3D, 
                 end: Point3D):
        self.start = start
        self.end = end


class Line3D:

    #cur_pos: np.array
    start: Point3D
    end: Point3D

    def __init__(self, *, 
                 start: Point3D, 
                 end: Point3D):
        if isinstance(start, np.array):
            start = Point3D(start)
        if isinstance(end, np.array):
            end = Point3D(end)
        self.start = start
        self.end = end

    def rotate(self, *, 
               center: Point3D, 
               R: Rotation):
        """ rotate self to get a new Line
        """
        new_start = Point3D(R.apply(self.start.coords - center.coords) + center.coords)
        new_end = Point3D(R.apply(self.end.coords - center.coords) + center.coords)
        return Line3D(start=new_start, end=new_end)
    
    def proj(self, z):
        """ Porject to a 2d line.
        """
        return Line2D(
            self.start.proj(z=z),
            self.end.proj(z=z),
        )

    def shift(self, v):
        """ Shift by v.
        """
        return Line3D(self.start.shift(v), self.end.shift(v))


output_dir = Path("/Users/zche/data/0xgenerator/space_travel/outputs/") 

def gen_eth():

    proj_z = 10

    n_frames = 100
    total_time = 10 # seconds
    output_size = np.array([600]*2)

    
    top     = Point3D((0, 0, 1))
    bottom  = Point3D((0, 0, -1))
    lines   = [
        Line3D(top,     (1, 0, 0)),
        Line3D(top,     (-1, 0, 0)),
        Line3D(top,     (0, 1, 0)),
        Line3D(top,     (0, -1, 0)),
        Line3D(bottom,  (1, 0, 0)),
        Line3D(bottom,  (-1, 0, 0)),
        Line3D(bottom,  (0, 1, 0)),
        Line3D(bottom,  (0, -1, 0)),
    ]

    center  = Point3D((0, 0, 0))
    lines = [_.shift(center.coords) for _ in lines]

    output_arrs = []


    angle_speed = pi * 2 # angle to rotate for a whole loop
    log.info("generating frames.....")
    bg_arr = frm.uniform(size=output_size, color=frm.BLACK)

    output_frames = []
    for i in tqdm(range(n_frames)):
        t = i / n_frames * total_time # seconds
        output_h, output_w = output_size
        xc, yc = output_h // 2, output_w // 2

        R = Rotation.from_rotvec(np.array([0, 0, 1]) * angle_speed * t)
        new_lines = [
            _.rotate(center=center, R=R)
            for _ in lines
        ]
        new_lines2d = [
            line.proj(z=proj_z) for line in new_lines
        ]

        new_im = io.np_to_im(bg_arr).convert("RGBA")
        drawer = ImageDraw.draw(new_im)

        for line2d in new_lines2d:
            drawer.line(list(line2d.start) + list(line2d.end), fill=0)
            
        output_frames.append(new_im)

    output_frames = []
    log.info("post processing frames.....")
    #for i, arr in tqdm(enumerate(output_arrs)):
    for i, frame in tqdm(enumerate(output_arrs)):
        #frame = io.np_to_im(arr, "RGB")
        output_frames.append(frame.quantize(kmeans=3))#dither=Image.NONE)
        enhancer = ImageEnhance.Contrast(frame)
        texture_im = enhancer.enhance(0.2)
        texture_im = texture_im.resize((500, 500), Image.NEAREST)

    output_file = output_dir / f"eth.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=5.0,
    )


parser = ArgumentParser()
#parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_eth()
