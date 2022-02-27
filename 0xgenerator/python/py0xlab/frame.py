from py0xlab import *
import numpy as np

def crop(arr, crop_limits=None, top_left=None, size=None):
    """ Return a sub image of arr.
    crop_limits = [
        left,
        top,
        right,
        bottom,
    ]
    right and bottom not included.
    """
    if crop_limits:
        return arr[
            crop_limits[0]:crop_limits[2],
            crop_limits[1]:crop_limits[3],
            :]
    elif top_left is not None and size is not None:
        return arr[
            top_left[0]:(top_left[0] + size[0]),
            top_left[1]:(top_left[1] + size[1]),
            :]
    else:
        raise ValueError("either crop_limits or (top_left, size) is required")


def get_coords(size):

    assert len(size) >= 2, "input must be a matrix or higher rank array."
    h, w = size[:2]
    x_coords = np.array([[i] * w for i in range(h)])
    y_coords = np.array([[j for j in range(w)]] * h)
    return x_coords, y_coords


def round_corner(arr, radius, bg_color):
    assert 0 <= radius < 0.5
    h, w, *_ = arr.shape
    x_coords, y_coords = get_coords(arr.shape)
    radius = min(h, w) * radius # convert radius to pixels
    inside_mask = np.array([[False] * w] * h)
    inside_mask = np.logical_or.reduce(
        (inside_mask, 
         np.logical_and(x_coords >= 0 + radius, x_coords <= h - radius - 1),
         np.logical_and(y_coords >= 0 + radius, y_coords <= w - radius - 1),
         np.logical_or.reduce(
                [
                    ((x_coords - xx) ** 2+(y_coords - yy) ** 2) <= radius ** 2 for xx, yy in (
                        (radius, radius),
                        (radius, w - radius),
                        (h - radius, radius),
                        (h - radius, w - radius)
                    )
                ]
         )))

    inside_mask = np.stack([inside_mask]*3, axis=2)
    bg = np.array([[bg_color]*w] * h)
    res = np.where(inside_mask,
                   arr,
                   bg
                   )
    return res



def get_bg_color(arr: np.array):
    
    assert arr.shape[0] > 2 and arr.shape[1] > 2, f"img too small or narrow: {arr.shape}"
    return arr[2, 2, :]


def is_same_color(arr, color, tol=10):
    """ Get a matrix of bools indicating each pixel has the same color as `color`.
    """
    color_diff = arr - np.array(color)
    max_color_diff = np.max(np.abs(color_diff), axis=2)
    mask = max_color_diff < tol
    return mask


def uniform(size, color):
    """ Create a white image.
    """
    return np.stack([
        np.ones(list(size)) * color[0],
        np.ones(list(size)) * color[1],
        np.ones(list(size)) * color[2],
    ], axis=2)

    
GREEN = np.array([100, 255, 0])
def green(size):
    return uniform(size, GREEN)

YELLOW = np.array([255, 215, 0])
def yellow(size):
    return uniform(size, YELLOW)

WHITE = np.array([255, 255, 255])
def white(size):
    """ Create a white image.
    """
    return uniform(size, WHITE)

BLACK = np.array([0, 0, 0])


def get_rgb(name, size=None):
    
    color = {
        #"pink": (230, 0, 122),
        "pink": (255, 0, 130),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "green": (100, 255, 0),
        "yellow": (255, 215, 0),
    }.get(name)
    if size is None:
        return color
    else:
        return uniform(size, color)
        

def where(mask, data1, data2):
    """ Replace the yellow in im1 by im2 if im2 is not white.
    - mask is a matrix
    - data1 is a w*h*3 array
    - data2 is a w*h*3 array
    """
    last_dim    = data1.shape[-1]
    mask        = np.stack([mask] * last_dim, axis=last_dim - 1)
    new_data    = np.where(mask, data1, data2)
    return new_data


def overlay(arr1, arr2):
    """ Overlay arr1 on top of arr2.
    Background of arr1 becomes transparent.
    """
    bg_color1 = get_bg_color(arr1)
    is_bg_color1 = is_same_color(arr1, bg_color1)
    return where(is_bg_color1, arr2, arr1)

    
def rotate(arr, k=None):
    """ Only support multiple of 90 degrees.
    """
    assert arr.shape[0] == arr.shape[1], f"only support square img, got shape {arr.shape}"
    if k is None:
        k = np.random.randint(0, 4)
    return np.rot90(arr, k, axes=(0, 1))


def replace_background_by_non_background(arr1, arr2, bg_color1=None, bg_color2=None):
    """ Replace the backgroud in arr1 by non-background in arr2.
    """
    assert arr1.shape == arr2.shape, f"inputs shape mismatch: {arr1.shape} vs {arr2.shape}"
    if bg_color1 is None:
        bg_color1   = arr1[5, 5, :]
    if bg_color2 is None:
        bg_color2   = arr2[3, 3, :]
    is_bg1      = is_same_color(arr1, bg_color1)
    is_bg2      = is_same_color(arr2, bg_color2)
    replace     = np.logical_and(is_bg1, np.logical_not(is_bg2))
    new_arr     = where(replace, arr2, arr1)
    return new_arr

    
def paste(arr1, arr2, loc, *, bg_color1, bg_color2):
    """ Paste arr1 as-is on arr2
    Position is ratio of size.
    """
    w1, h1, *_ = arr1.shape
    w2, h2, *_ = arr2.shape
    x, y = loc

    offset_x = w2 + w1 - 1
    offset_y = h2 + h1 - 1
    if x >= w2:
        x -= offset_x
        return paste(arr1, arr2, (x, y), bg_color1=bg_color1, bg_color2=bg_color2)
    elif y >= h2:
        y -= offset_y
        return paste(arr1, arr2, (x, y), bg_color1=bg_color1, bg_color2=bg_color2)
    elif x + w1 <= 0:
        x += offset_x
        return paste(arr1, arr2, (x, y), bg_color1=bg_color1, bg_color2=bg_color2)
    elif y + h1 <= 0:
        y += offset_y
        return paste(arr1, arr2, (x, y), bg_color1=bg_color1, bg_color2=bg_color2)
    else:
        l2 = max(x, 0)
        t2 = max(y, 0)
        r2 = min(x + w1, w2)
        b2 = min(y + h1, h2)

        l1 = l2 - x
        t1 = t2 - y
        r1 = r2 - l2 + l1
        b1 = b2 - t2 + t1
        #log.info([l2, t2, r2, b2])
        #log.info([l1, t1, r1, b1])
    
        res = np.array(arr2)
        #res[l2:r2, t2:b2] = arr1[l1:r1, t1:b1]
        res[l2:r2, t2:b2] = replace_background_by_non_background(
            res[l2:r2, t2:b2],
            arr1[l1:r1, t1:b1],
            bg_color1,
            bg_color2,
        )
        return res
    
    
if __name__ == "__main__":

    arr1 = np.ones((3, 4, 3))
    arr2 = np.zeros((5, 6, 3))
    log.info(
        (
            np.sum(paste(arr1, arr2, loc=[0, 0]), axis=2),
            np.sum(paste(arr1, arr2, loc=[-1, -1]), axis=2),
            np.sum(paste(arr1, arr2, loc=[3, 3]), axis=2),
        )
    )