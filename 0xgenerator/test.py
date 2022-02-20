import numpy as np
from pathlib import Path
from PIL import Image
import logging as log
from tqdm import tqdm
from PIL import ImageFilter
from argparse import ArgumentParser
log.basicConfig(level=log.DEBUG)


def compile(frames, *, output_file, duration):

    frame_one = frames[0]
    frame_one.save(output_file,
                   format="GIF", 
                   append_images=frames[1:],
                   save_all=True, 
                   duration=duration, 
                   loop=0)
    log.info(output_file)


def read_gif(path, *, samples, size, rescale=1.0):
    
    gif_image = Image.open(path)
    n_frames = gif_image.n_frames
    log.info(f"total number of frames = {n_frames}")

    #for i in tqdm(range(n_frames)):
    x_0, y_0 = gif_image.size[0] // 2, gif_image.size[1] // 2
    tmp = int((min(x_0, y_0) - 1) * rescale)
    crop_limits = (x_0 - tmp, y_0 - tmp, x_0 + tmp, y_0 + tmp)
    log.info(f"cropped at: {crop_limits}")
    
    frames = []
    for i in range(n_frames):
        gif_image.seek(i)
        frame = gif_image.convert("RGB")
        frame = frame.crop(crop_limits)
        frame = frame.resize(size, Image.ANTIALIAS)
        frames.append(frame)
    
    if n_frames < samples:
        log.info(f"number of frames < samples={samples}; will keep all frames")
    else:
        idxs = [int(n_frames / samples * i) for i in range(samples)]
        log.info(f"taking {idxs}-th frames")
        frames = [frames[idx] for idx in idxs]

    return frames

def replace_yellow_by_non_white(data1, data2):
    """ Replace the yellow in im1 by im2 if im2 is not white.
    """
    #yellow  = (255, 215, 0)
    yellow  = data1[5, 5, :]
    white   = (255, 255, 255)
    is_yellow   = (np.max(np.abs(data1 - np.array(yellow)), axis=2) <  10)
    not_white   = (np.max(np.abs(data2 - np.array(white )), axis=2) >= 30)
    replace     = np.stack([np.logical_and(is_yellow, not_white)] * 3, axis=2)
    new_data    = np.where(replace, data2, data1)
    return new_data

def main(fg_file, input_dir, output_file):

    n_frames    = 60
    size        = (500, 500)
    fg_frame    = Image.open(fg_file)
    fg_frame    = fg_frame.resize(size).convert("RGB")
    bg_frames   = read_gif(f"{input_dir}/sakura.gif", samples=n_frames, size=fg_frame.size, rescale=0.8)

    # translate sequence of label to sequence of frames
    frames = []
    im_data1 = np.array(fg_frame)
    for bg_frame in tqdm(bg_frames):
        im_data2 = np.array(bg_frame)
        new_data = replace_yellow_by_non_white(im_data1, im_data2)
        frame = Image.fromarray(new_data.astype('uint8'), "RGB")
        frame = frame.filter(ImageFilter.SMOOTH)
        frames.append(frame)

    # set duration of each frame according total time and number of frames
    total_time = 5000
    duration = total_time / len(frames)
    compile(frames, output_file=output_file, duration=duration)

parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    root_dir = "/Users/zche/data/0xgenerator/test0"
    input_dir = f"{root_dir}/inputs"
    output_dir = f"{root_dir}/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    #fg_file = f"{input_dir}/p1.jpg"

    #file_name = "p4" # file name without suffix
    file_name = parser.parse_args().file_name
    fg_file = f"{input_dir}/{file_name}.png"
    main(fg_file, input_dir, f"{output_dir}/{file_name}_.gif")