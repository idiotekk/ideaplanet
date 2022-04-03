from mimetypes import init
from re import X
from numpy.core import multiarray
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

import numpy as np
import sys
import cv2

class Vertex:

    # quantities below are all normalized to 0
    _p: np.ndarray # current position
    _v: np.ndarray # vilocity
    _t: float # current t
    _a: np.ndarray # accecelration ( delta speed )
    _edges: list = []

    def __init__(self, *, p, v, t=0):
        assert len(p) == 2 and len(v) == 2, "p and v must be 2-d"
        self._p = np.array(p)
        self._v = np.array(v)
        self._t = t
        self._color = [np.random.randint(100, 200) for _ in range(3)]

    @property
    def x(self):
        return self._p[0]

    @property
    def y(self):
        return self._p[1]

    @property
    def p(self):
        return self._p

    def update(self, dt, a):
        """ Move to new position.
        """
        self._t += dt
        self._a = a
        self._v += self._a * dt
        self._p += self._v * dt
        self._p = self._p % 1 # circulate

    def absp(self, shape): 
        # convert from rel_pos to abs_pos (integer!!!)
        h, w, *_ = shape
        return (np.array([h, w]) * self._p).astype(int)

    def render(self, canvas):
        h, w, *_ = canvas.shape
        absp = self.absp(canvas.shape)
        r = int(0.01 * min(h, w))
        return cv2.circle(canvas, absp, r, self._color, r*2)
        

class Edge:
    
    _s: Vertex # starting
    _e: Vertex # ending
    _t: float # spawn time
    _thick: float

    def __init__(self, s, e, t=0) -> None:
        self._s = s
        self._s = e

    @property
    def length(self):
        return dist(self._s, self._e)
        
    def render(self, canvas):
        shape = canvas.shape
        color = (255 ,255, 255) * self.length
        return cv2.line(canvas, 
                        self._p.absp(shape),
                        self._q.absp(shape),
                        color, 
                        self.thick
                        )

    def thick(self):
        return 2

        
def dist(a: Vertex, b: Vertex):
    return np.sqrt((a.p - b.p) ** 2)


        
class Pool:
    
    _vertices = []
    _pos_matrix : np.ndarray

    def __init__(self) -> None:
        return

    def update(self):
        self._pos_matrix = np.stack([v.p for v in self._vertices], axis=1)
        log.info(f"refreshesh pos matrix... shape = {self._pos_matrix.shape}")

    def add_vertex(self, v: Vertex):
        self._vertices.append(v)

    @property
    def pos_matrix(self):
        return self._pos_matrix


out_dir = Path("/Users/zche/data/0xgenerator/generative/outputs/") 


def speed_generator(scalar=1.0):

    absv = np.random.randn() * scalar
    angle = np.random.randn() * pi * 2
    return np.array([
        np.cos(angle),
        np.sin(angle)
    ]) * absv

def gen_graph():

    n_vertices = 20
    pool = Pool()

    for _ in tqdm(range(n_vertices)):
        x = np.random.rand()
        y = np.random.rand()
        v = speed_generator(scalar=0.2)
        new_vertex = Vertex(p=(x, y), v=v, t=0)
        pool.add_vertex(new_vertex)

    edges = []
    for i, vi in enumerate(pool._vertices):
        for j, vj in enumerate(pool._vertices):
            edges.append(Edge(vi, vj))

    out_shape = (300, 300)
    out_h, out_w = out_shape

    n_frames = 100
    output_arrs = []

    log.info("generating frames.....")
    for _ in tqdm(range(n_frames)):
        new_arr = np.zeros((out_h, out_w, 3), np.uint8)
        dt  = 1 / n_frames
        pool.update()
        for i, v in enumerate(pool._vertices):
            a = pool.calc_force(j)
            v.update(dt, a=a)
            new_arr = v.render(new_arr)
        for k, e in enumerate(edges):
            if e.length < 0.3:
                e.render(new_arr)
        output_arrs.append(new_arr)

    output_frames = []
    log.info("post processing frames.....")
    for i, arr in tqdm(enumerate(output_arrs)):
        frame = io.np_to_im(arr, "RGB")
        output_frames.append(frame.quantize(kmeans=3))
        #enhancer = ImageEnhance.Contrast(frame)
        #texture_im = enhancer.enhance(0.2)

    output_frames = output_frames[(n_frames//2):]
    output_frames = output_frames + output_frames[::-1]

    output_file = out_dir / f"graph.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=3.0,
    )

    return
    videodims = (100,100)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')    
    video = cv2.VideoWriter(str("graph.mp4"),fourcc, 60,videodims)
    img = Image.new('RGB', videodims, color = 'darkred')
    #draw stuff that goes on every frame here
    for i in range(0,60*60):
        imtemp = img.copy()
        # draw frame specific stuff here.
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()

parser = ArgumentParser()
#parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_graph()