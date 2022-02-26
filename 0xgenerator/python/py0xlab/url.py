""" This lib is for 
"""
from py0xlab import *
import numpy as np

import os
import requests
from selenium import webdriver
# the 2 lines below are for printing logging messages 
import logging as log
log.basicConfig(level=log.INFO)
# the 3 lines below make browser invisible
from selenium.webdriver.chrome.options import Options




def download_nft(out_dir: Path):

    collection_addr = "0x2eb6be120ef111553f768fcd509b6368e82d1661"
    item_id = 6406 # the number of the nft in the collection
    item_page = f"https://opensea.io/assets/{collection_addr}/{item_id}#main"
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    with webdriver.Chrome(options=chrome_options) as driver: # open a browser 

        # step1: open the page and find the link to embedded player
        log.info(f"opening webpage: {item_page}")
        driver.get(item_page)
        #elements = driver.find_elements_by_class_name("Image--image")

        #driver.implicitly_wait(10)
        elements = driver.find_elements_by_tag_name("img")
        #elements = driver.find_elements_by_id("__next")
        #elements = driver.find_elements_by_tag_name("body")
        #elements = driver.find_elements_by_class_name("item--small")
        if not elements:
            raise ValueError("no elements found")
        log.info(f"{len(elements)} elements found")
        for i, elem in enumerate(elements):
            type_ = elem.get_attribute("class")
            log.info(("---------", i, type_))

        return
        elements = driver.find_elements_by_class_name("Image--image")
        for i, elem in enumerate(elements):
            image_url = elem.get_attribute("src")
            file_ = requests.get(image_url)
            out_file = out_dir / str(item_id) / f"item{i}.png"
            log.info(f"about to download image to: {out_file}")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                f.write(file_.content)
            log.info(f"finished downloading image to: {out_file}")

        """
        # step2: open the embedded player and find the link to the mp3
        log.info(f"redirecting to: {player_url}")
        driver.get(player_url)
        for elem in driver.find_elements_by_class_name("Image--image"):
            href = elem.get_attribute("href")
            if href and "mp3" in href:
                audio_url = href # this is the link to the mp3
        """

        

#parser = ArgumentParser()
#parser.add_argument("file_name", type=str)
    
if __name__ == "__main__":

    out_dir = Path("/Users/zche/data/0xgenerator/sakura_rain/inputs/")
    download_nft(out_dir)