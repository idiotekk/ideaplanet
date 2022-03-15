from asyncio import start_unix_server
import sys, os
import pprint
from tokenize import blank_re
import numpy as np
from PIL import ImageFilter, ImageDraw
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
pprint.pprint(sys.path)
from py0xlab import *
from py0xlab.frame import where, is_same_color, get_bg_color
from py0xlab import frame as frm
from py0xlab import io
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
log.basicConfig(level=log.DEBUG)

class Drop:

    start: np.array
    angle: float
    length: float
    width: float

    def __init__(self, *, start, angle, length, width):
        """ ...
        """
        self.start = start
        self.angle = angle
        self.length = length
        self.width = width

    def render(self, line_drawer):
          
        # create line image
        #line_drawer = ImageDraw.Draw(img)  
        shape = [self.start, self.start + self.length * np.array(
            [np.cos(self.angle), np.sin(self.angle)]
        )]
        line_drawer.line(shape, fill ="none", width = self.width)


def rain_drop(file_name):

    #input_file = Path(os.path.expandvars("~/cloud/data/0xgenerator/inputs")) / "sakura.png"
    input_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/inputs/")
    output_dir = Path("/Users/zche/data/0xgenerator/rain/ouputs/") 

    origin_size = 1200
    output_size = (400, 400)
    out_w, out_h = output_size
    fg_frame    = io.read_frame(input_dir / f"{file_name}", size=output_size)


    n_drops = 100
    drops = []
    for i in tqdm(range(n_drops)):

        #if i < 10: # for debug
            #io.np_to_im(p.arr).save(str(output_dir / f"test_{i}.png"))
        d = Drop(start=np.array([
                np.random.randint(0, out_h),
                np.random.randint(0, out_w),
            ]),
                 angle=0,
                 length=out_h // 5,
                 width=3)
        drops.append(d)

    n_frames = 150
    output_frames = []

    #blank_bg = frm.yellow(output_size)
    for t in tqdm(range(n_frames)):
        new_im = fg_frame.copy()
        rain_drawer = ImageDraw.Draw(new_im)
        for d in drops:
            d.render(rain_drawer)
        output_frames.append(new_im)
        output_frames[i] = new_im.quantize(kmeans=3)

    output_file = output_dir / f"rain_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=7.0,
    )
    


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_sakura_gif(file_name=parser.parse_args().file_name)
