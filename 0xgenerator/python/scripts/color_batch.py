from asyncio import start_unix_server
import sys, os
import pprint
import numpy as np
from PIL import ImageFilter
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
pprint.pprint(sys.path)
from py0xlab import *
from py0xlab.frame import where, is_same_color, get_bg_color
from py0xlab import frame as frm
from py0xlab import io
from pathlib import Path
from PIL import ImageEnhance
from tqdm import tqdm
import numpy as np
import pandas as pd
from argparse import ArgumentParser
log.basicConfig(level=log.DEBUG)

import cv2
import matplotlib.pyplot as plt

input_dir = Path("/Users/zche/cloud/data/0xgenerator/andy/inputs/")
output_dir = Path("/Users/zche/cloud/data/0xgenerator/andy/outputs/") 

    
def main(file_name):

    
    image = cv2.imread(str(input_dir / f"{file_name}"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    print(pixel_values.shape)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 9
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image)
    plt.show()


    return
    input_im = io.read_frame(input_dir / f"{file_name}", to_np=False) 
    #input_im = input_im.resize((300, 300))
    input_w, input_h = input_im.size
    input_arr = np.array(input_im)

    colors_found = [
        input_arr[i,j,:] for i in range(input_h) for j in range(input_w)
    ]
    unique_colors = np.unique(colors_found)

    input_im_q = input_im.quantize(kmeans=10)
    input_arr_q = np.array(input_im_q)
    colors_found = [
        input_arr_q[i,j] for i in range(input_h) for j in range(input_w)
    ]
    unique_colors_q = pd.Series(colors_found).value_counts()
    log.info({
        "n unique colors": len(unique_colors),
        "unique colors q": unique_colors_q,
    })

    return

    

    output_file = Path(output_dir) / f"{file_name.split('.')[0]}_4by4.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    io.save_im(output_im, output_file)
    output_im.show()


parser = ArgumentParser()
parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    main(file_name=parser.parse_args().file_name)
