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

    
def get_bg_color(arr: np.array):
    
    assert arr.shape[0] > 2 and arr.shape[1] > 2, f"img too small or narrow: {arr.shape}"
    return arr[2, 2, :]


def is_same_color(arr, color, tol=10):
    """ Get a matrix of bools indicating each pixel has the same color as `color`.
    """
    return np.max(np.abs(arr - np.array(color)), axis=2) <  10



def uniform(size, color):
    """ Create a white image.
    """
    return np.stack([
        np.ones(list(size)) * color[0],
        np.ones(list(size)) * color[1],
        np.ones(list(size)) * color[2],
    ], axis=2)

    
GREEN = (100, 255, 0)
def green(size):
    return uniform(size, GREEN)


WHITE = (255, 255, 255)
def white(size):
    """ Create a white image.
    """
    return uniform(size, WHITE)


def where(mask, data1, data2):
    """ Replace the yellow in im1 by im2 if im2 is not white.
    - mask is a matrix
    - data1 is a w*h*3 array
    - data2 is a w*h*3 array
    """
    last_dim    = data1.shape[-1]
    mask        = np.stack([mask] * last_dim, axis=last_dim - 1)
    new_data    = np.where(mask, data2, data1)
    return new_data


def overlay(arr1, arr2):
    """ Overlay arr1 on top of arr2.
    Background of arr1 becomes transparent.
    """
    bg_color1 = get_bg_color(arr1)
    is_bg_color1 = is_same_color(arr1, bg_color1)
    return where(is_bg_color1, arr2, arr1)


def replace_background_by_non_background(arr1, arr2):
    """ Replace the backgrou in arr1 by non-background in arr2.
    """
    bg_color1   = arr1[5, 5, :]
    bg_color2   = arr2[3, 3, :]
    is_bg1      = (np.max(np.abs(arr1 - np.array(bg_color1)), axis=2) <  10)
    is_bg2      = (np.max(np.abs(arr2 - np.array(bg_color2)), axis=2) >= 30)
    replace     = np.logical_and(is_bg1, np.logical_not(is_bg2))
    replace     = np.stack([replace] * 3, axis=2)
    new_arr    = np.where(replace, arr2, arr1)
    return new_arr