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

class Piece:

    arr: np.array
    speed_x: float
    speed_y: float
    start_x: float
    start_y: float

    def __init__(self):
        pass


def gen_sakura_gif(file_name):

    #input_file = Path(os.path.expandvars("~/cloud/data/0xgenerator/inputs")) / "sakura.png"
    input_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/inputs/")
    output_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") 

    origin_size = 1200
    output_size = (400, 400)
    fg_frame = io.read_frame(input_dir / f"{file_name}", size=output_size)
    #skr_frame = io.read_frame(input_dir / "sakura.png", to_np=False) # sakura frame
    skr_frame = io.read_frame(input_dir / "cats.png", to_np=False) # sakura frame
    skr_frame.resize((100, 100))

    w, h = skr_frame.size
    nn = 4
    raw_piece_size = w // nn, h // nn 
    raw_pieces = []
    for i in range(nn):
        for j in range(nn):
            raw_piece = skr_frame.crop((
                w // nn * i,
                h // nn * j,
                w // nn * i + raw_piece_size[0],
                h // nn * j + raw_piece_size[1]
            ))
            raw_pieces.append(raw_piece)


    pieces = []
    
    base_speed = 3.2

    n_pieces = 101
    for i in tqdm(range(n_pieces)):
        rescale = (output_size[0] / origin_size)  
        #distance_factor = np.random.rand() 
        distance_factor = (n_pieces - i) / n_pieces
        rescale_idio =  1.8 #(0.2 + 0.8 * distance_factor) * 4
        piece_idx = np.random.randint(0, len(raw_pieces) - 1)
        piece_size = [int(_ * rescale * rescale_idio) for _ in raw_piece_size]
        piece = raw_pieces[piece_idx].resize(piece_size) # piece to place on top of background rescaled
        #piece.save(str(output_dir / f"sakura_piece_rescaled.png")) # for debug
        #log.info(f"piece {piece_idx} rescaled by {rescale} to new size {piece_size}")
        piece_arr = np.array(piece)
        replace_ = is_same_color(piece_arr, get_bg_color(piece_arr))
        piece_arr = where(
            replace_,
            frm.yellow(piece_size),
            piece_arr) # replace background by green
        #piece_arr = frm.rotate(piece_arr, k=np.random.randint(0, 4)) not rotate cat
        
        p = Piece()
        p.arr = piece_arr
        p.speed_x = base_speed * (0.1 + 0.5 * distance_factor + 0.5*np.random.rand()) # horizontal speed at each frame, how much of the frame does the p fly for the whole time
        #p.speed_y = -(1+np.random.rand()) / 2 * p.speed_x 
        p.speed_y = -(1) / 2 * p.speed_x 
        #p.start_x = np.random.rand()
        #p.start_y = np.random.rand()
        p.start_x = np.mod(i * 19, n_pieces) / n_pieces
        p.start_y = np.mod(i * 13, n_pieces) / n_pieces

        #if i < 10: # for debug
            #io.np_to_im(p.arr).save(str(output_dir / f"test_{i}.png"))
        pieces.append(p)

    out_w, out_h = output_size
    n_frames = 150
    output_frames = []

    blank_bg = frm.yellow(output_size)
    for t in tqdm(range(n_frames + 100)):
        if t < 100:
            continue
        new_bg = np.array(blank_bg)
        for p in pieces:
            loc = (
                int((p.speed_x / n_frames * t + p.start_x) * out_w), 
                int((p.speed_y / n_frames * t + p.start_y) * out_h)
            )
            new_bg = frm.paste(p.arr, new_bg, loc=loc, bg_color1=frm.YELLOW, bg_color2=frm.YELLOW)
        arr = frm.replace_background_by_non_background(
            fg_frame, new_bg,
            bg_color1=get_bg_color(fg_frame),
            bg_color2=frm.YELLOW,
        )
        output_frames.append(arr)
    
    for i, arr in tqdm(enumerate(output_frames)):
        frame = io.np_to_im(arr, "RGB")
        #frame = frame.convert("YCbCr")
        #output_frames[i] = frame
        output_frames[i] = frame.quantize(dither=Image.NONE)

    #output_file = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") / f"sakura_{file_name.split('.')[0]}.gif"
    output_file = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") / f"cat_{file_name.split('.')[0]}.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=6.0,
    )
    


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_sakura_gif(file_name=parser.parse_args().file_name)
