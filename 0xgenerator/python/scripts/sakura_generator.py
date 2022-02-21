import sys, os
import pprint
pprint.pprint(sys.path)
import numpy as np
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
from py0xlab import *
from py0xlab.frame import where, is_same_color, get_bg_color
from py0xlab import frame, io
from pathlib import Path
from tqdm import tqdm
log.basicConfig(level=log.DEBUG)


def gen_sakura_gif():

    input_file = Path(os.path.expandvars("~/cloud/data/0xgenerator/inputs")) / "sakura.png"
    #arr = read_frame(input_file)
    frame = io.read_frame(input_file, to_np=False)

    w, h, _ = frame.size
    nn = 4
    raw_piece_size = w // nn, h // nn 
    raw_pieces = []
    for i in range(nn):
        for j in range(nn):
            raw_pieces.append(
                #crop(arr,
                #     top_left=(w // nn * i, w // nn * j),
                #     size=raw_piece_size
                #)
                frame.crop(
                    w // nn * i,
                    h // nn * j,
                    w // nn * i + raw_piece_size[0],
                    h // nn * j + raw_piece_size[1]
                )
            )

    rescale = 1.0
    piece_idx = 0
    piece_size = [int(_ * rescale) for _ in raw_piece_size]
    piece = raw_pieces[piece_idx].resize(piece_size) # piece to place on top of background rescaled
    log.info(f"piece {piece_idx} rescaled by {rescale} to new size {piece_size}")
    piece_arr = np.array(piece)
    piece_arr = where(
        is_same_color(piece_arr, get_bg_color(piece_arr)),
        frame.green(piece_size),
        piece_arr) # replace background by green

    speed_x = 0.1 # horizontal speed at each frame, how much of the frame size does the piece fly though
    speed_y = 0.1 # vertical speed
    output_size = [500, 500]
    n_frames = 10
    output_frames = []
    for t in range(n_frames):
        new_arr = frame.green(output_size)
        top_left = speed_x * t, speed_y * t
        new_arr[
            top_left[0]: piece_size[0],
            top_left[1]: piece_size[1],
            :] = piece_arr
        output_frames.append(
            frame.np_to_im(new_arr)
        )

    output_file = Path(os.path.expandvars("~/cloud/data/0xgenerator/outputs")) / "sakura.gif"
    io.compile_gif(
        output_frames,
        output_file=output_file,
        total_time=1.0,
    )
    

parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    gen_sakura_gif()
