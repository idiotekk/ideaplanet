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
log.basicConfig(level=log.DEBUG)


def gen_lightning_gif(file_name):

    #input_file = Path(os.path.expandvars("~/cloud/data/0xgenerator/inputs")) / "sakura.png"
    input_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/inputs/")
    output_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") 

    origin_size = 1200
    output_size = (400, 400)
    fg_frame = io.read_frame(input_dir / f"{file_name}", size=output_size)
    skr_frame = io.read_frame(input_dir / "lightning.png", to_np=False) # sakura frame
    skr_frame.resize(output_size)

    h, w = output_size


    x0, y0 = 0, w // 2

    canvas = frm.uniform(size=output_size, color=frm.BLACK)

    output_arrs = []
    xt, yt = x0, y0
    vx, vy = 10, 10
    for t in range(1000):
        rt = np.random.rand() * r0
        thetat = np.random.rand() * pi * 2
        x_next = xt + rt*np.cos(thetat) + vx
        y_next = yt + rt*np.sin(thetat) + vy


    

    output_file = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") / f"sakura_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=7.0,
    )
    


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_lightning_gif(file_name=parser.parse_args().file_name)
