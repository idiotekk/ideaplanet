from 0xgenerator import log


def compile(frames, *, output_file, duration):

    frame_one = frames[0]
    frame_one.save(output_file,
                   format="GIF", 
                   append_images=frames[1:],
                   save_all=True, 
                   duration=duration, 
                   loop=0)
    log.info(output_file)