from py0xlab import *
import numpy as np
import cv2, os
import pickle
import hashlib
os.environ["PYTHONHASHSEED"] = "0"


def my_hash(arr):
    return hashlib.sha1(arr).hexdigest()

def get_cache_file(hs):
    file_ = f"/tmp/{hs}"
    return file_

def cached(hs):
    file_ = get_cache_file(hs)
    return os.path.exists(file_)
    

#def clusterize(input_file, k=5):
def clusterize(input_arr, k=5, cache=True):

    #image = cv2.imread(str(input_file))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if cache is True:
        hash_value = my_hash(input_arr.astype(int) + k)
        cache_file = get_cache_file(hash_value)
        if cached(hash_value):
            log.info(f"reading cached result from {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    res = pickle.load(f)
                    return res
            except Exception:
                log.info(f"failed reading cached result from {cache_file}; will do clustering...")
        else:
            log.info(f"cached file {cache_file} not found; will do clustering...")


    image = input_arr
    input_h, input_w = image.shape[:2]
    log.info({"height": input_h, "width": input_w})
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    log.info(f"clusterizing, k = {k}")
    compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    log.info(f"done clusterizing, k = {k}; extracting")
    # convert back to 8 bit values
    centers = np.uint8(centers)
    assert len(centers) == k, f"sanity check: number of centers must be k = {k}"
    log.info(f"centers: {centers}")
    # flatten the labels array
    labels = labels.flatten()
    centers_and_bg = np.row_stack([centers, np.array([[255, 255, 255]])])
    print(centers.shape, centers_and_bg.shape)
    # convert all pixels to the color of the centroids
    #centers = centers[np.random.permutation(centers.shape[0]), :]
    segmented_image = centers[labels]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    labels_reshaped = labels.reshape(image.shape[:2])

    res = segmented_image, labels_reshaped

    if cache is True:
        with open(cache_file, "wb") as f:
            log.info(f"caching result to {cache_file}")
            pickle.dump(res, f)

    return res