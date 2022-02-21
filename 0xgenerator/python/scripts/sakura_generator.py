import sys, os
import pprint
import numpy as np
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


def gen_sakura_gif():

    #input_file = Path(os.path.expandvars("~/cloud/data/0xgenerator/inputs")) / "sakura.png"
    input_dir = Path("/Users/zche/data/0xgenerator/test0/inputs/")
    output_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") 

    fg_frame = io.read_frame(input_dir / "p4.png")
    skr_frame = io.read_frame(input_dir / "sakura.png", to_np=False) # sakura frame

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

    rescale = 0.2
    piece_idx = 0
    piece_size = [int(_ * rescale) for _ in raw_piece_size]
    piece = raw_pieces[piece_idx].resize(piece_size) # piece to place on top of background rescaled
    piece.save(str(output_dir / f"sakura_piece_rescaled.png"))
    log.info(f"piece {piece_idx} rescaled by {rescale} to new size {piece_size}")
    piece_arr = np.array(piece)
    replace_ = is_same_color(piece_arr, get_bg_color(piece_arr))
    piece_arr = where(
        replace_,
        frm.green(piece_size),
        piece_arr) # replace background by green

    speed_x = 0.1 # horizontal speed at each frame, how much of the frame size does the piece fly though
    speed_y = 0.1 # vertical speed
    output_size = fg_frame.shape[:2]
    out_w, out_h = output_size
    n_frames = 10
    output_frames = []

    for t in range(n_frames):
        new_bg = frm.green(output_size)
        top_left = int(speed_x * t * out_w), int(speed_y * t * out_h)
        new_bg[
            top_left[0]: top_left[0] + piece_size[0],
            top_left[1]: top_left[1] + piece_size[1],
            :] = piece_arr
        new_arr = frm.replace_background_by_non_background(
            fg_frame, new_bg
        )
        output_frames.append(
            io.np_to_im(new_arr)
        )

    output_file = Path("/Users/zche/data/0xgenerator/sakura_rain/ouputs/") / "sakura.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=1.0,
    )
    


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_sakura_gif()
