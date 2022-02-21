from py0xlab import *
import numpy as np


def np_to_im(arr):
    
    return Image.fromarray(arr.astype('uint8'), "RGB")


def compile_gif(frames, *, output_file, total_time):
    """ Compile frame into gif, same as output_file.
    The frames will be evenly executed within `total_time` (unit=seconds)
    """
    duration = total_time * 1000.0 / len(frames)
    frame_one = frames[0]
    log.info({"number of frames": len(frames),
              "total_time": total_time,
              "duraion each frame (ms)": duration})
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    frame_one.save(str(output_file),
                   format="GIF", 
                   append_images=frames[1:],
                   save_all=True, 
                   duration=duration, 
                   loop=0)
    log.info(output_file)


def read_gif(path, *, samples, size, rescale=1.0, to_np=True, to_rgb=True):
    """ Read gif as a list of frames.
    """
    
    gif_image = Image.open(str(path))
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
        frame = frame.crop(crop_limits)
        frame = frame.resize(size, Image.ANTIALIAS)
        if to_rgb is True:
            frame = gif_image.convert("RGB")
        if to_np is True:
            frame = np.array(frame)
        frames.append(frame)
    
    if n_frames < samples:
        log.info(f"number of frames < samples={samples}; will keep all frames")
    else:
        idxs = [int(n_frames / samples * i) for i in range(samples)]
        log.info(f"taking {idxs}-th frames")
        frames = [frames[idx] for idx in idxs]

    return frames

    
def read_frame(path, *, to_np=True, to_rgb=True):
    """ Read a single frame from file (jpeg, png, etc.).
    """
    frame = Image.open(str(path))
    if to_rgb is True:
        frame = frame.convert("RGB")
    if to_np is True:
        frame = np.array(frame)