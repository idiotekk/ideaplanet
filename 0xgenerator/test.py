from codecs import BOM_UTF16_BE
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import logging as log
from tqdm import tqdm
log.basicConfig(level=log.DEBUG)

#    pics = glob.glob(f"{input_dir}/*.png")
#    frames = [Image.open(_) for _ in pics]
#    frame_one = frames[0]
#    output_file = f"{output_dir}/test0.gif" 
#    frame_one.save(output_file,
#                   format="GIF", 
#                   append_images=frames[1:],
#                   save_all=True, 
#                   duration=500, 
#                   loop=1)
#    log.info(output_file)


def open_frames(files, input_dir, suffix=".jpg"):
    return {
        _ : Image.open(f"{input_dir}/{_}{suffix}") for _ in files
    }


def compile(frames, *, output_file, duration):

    frame_one = frames[0]
    frame_one.save(output_file,
                   format="GIF", 
                   append_images=frames[1:],
                   save_all=True, 
                   duration=duration, 
                   loop=0)
    log.info(output_file)


def same_color(x, y, tol=2):
    """ True if x and y are almost the same color.
    """
    return all(abs(i-j)<= tol for i, j in zip(x, y))


def yellow_to_black(im):
    newimdata = []
    yellow = (255, 215, 0)
    blackcolor = (0,0,0)
    for color in im.getdata():
        if same_color(color, yellow, tol=30):
            newimdata.append(blackcolor)
        else:
            newimdata.append(color)
    newim = Image.new(im.mode, im.size)
    newim.putdata(newimdata)
    return newim



def read_gif(path, *, samples, resize, crop=1.0):
    
    log.info(f"about to read {path}")
    gif_image = Image.open(path)
    log.info(f"finished reading {path}")

    n_frames = gif_image.n_frames
    log.info(f"total number of frames = {n_frames}")
    frames = []
    log.info(f"extracting frames from gif")

    #for i in tqdm(range(n_frames)):
    x_0, y_0 = gif_image.size[0] // 2, gif_image.size[1] // 2
    tmp = int((min(x_0, y_0) - 1) * crop)
    crop_limits = (x_0 - tmp, y_0 - tmp, x_0 + tmp, y_0 + tmp)
    log.info(f"cropped at: {crop_limits}")
    
    for i in range(n_frames):
        gif_image.seek(i)
        frame = gif_image.convert("RGB")
        frame = frame.crop(crop_limits)
        frame = frame.resize(resize, Image.ANTIALIAS)
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
    newimdata = []
    yellow = (255, 215, 0)
    white = (255, 255, 255)
    is_yellow = (np.max(np.abs(data1 - np.array(yellow)), axis=2) <  10)
    not_white = (np.max(np.abs(data2 - np.array(white )), axis=2) >= 30)
    replace = np.logical_and(is_yellow, not_white)
    replace = np.stack([replace]*3, axis=2)
    new_data = np.where(replace, data2, data1)
    return new_data


def main(input_dir, output_file):

    n_frames = 200
    #fg_frames = open_frames(["p1", "p2", "p3"], input_dir, suffix=".jpg")
    fg_frame = Image.open(f"{input_dir}/p1.jpg")
    w, h = fg_frame.size
    bg_frames = read_gif(f"{input_dir}/sakura.gif", samples=n_frames, resize=(w, h))

    # translate sequence of label to sequence of frames
    frames = []
    im_data1 = np.array(fg_frame)
    for i, bg_frame in tqdm(enumerate(bg_frames)):
        im_data2 = np.array(bg_frame)
        new_data = replace_yellow_by_non_white(im_data1, im_data2)
        frame = Image.fromarray(new_data.astype('uint8'), "RGB")
        frames.append(frame)

    # set duration of each frame according total time and number of frames
    total_time = 10000
    duration = total_time / len(frames)
    log.info(duration)

    # compile
    compile(frames, output_file=output_file, duration=duration)
    
if __name__ == "__main__":

    root_dir = "/Users/zche/data/0xgenerator/test0"
    input_dir = f"{root_dir}/inputs"
    output_dir = f"{root_dir}/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main(input_dir, f"{output_dir}/sakura.gif")