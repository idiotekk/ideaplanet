"""
import requests
#url = "https://api.opensea.io/api/v1/collections?asset_owner=0xE5d4924413ae59AE717358526bbe11BB4A5D76b9&offset=0&limit=300"
#url = "https://opensea.io/collection/azuki"
url = "https://opensea.io/assets/0x2eb6be120ef111553f768fcd509b6368e82d1661/1884"
headers = {"Accept": "application/json"}
response = requests.request("GET", url, headers=headers)
print(response.text)
"""

from argparse import ArgumentParser
from ast import arg
import os
import pathlib
import requests
from selenium import webdriver
import datetime
# the 2 lines below are for printing logging messages 
import logging as log
log.basicConfig(level=log.INFO)
# the 3 lines below make browser invisible
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import utils
from utils import Timer
chrome_options = utils.get_chrome_options()

base_url_map = {
    "azuki": "https://opensea.io/assets/0xed5af388653567af2f388e6224dc7c4b3241c544/",
    "0xzuki": "https://opensea.io/assets/0x2eb6be120ef111553f768fcd509b6368e82d1661/",
}

def get_chrome_options():

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    #chrome_options.add_argument("--window-size=1920,1080")
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'    
    chrome_options.add_argument('user-agent={0}'.format(user_agent))
    return chrome_options

def download_img(url=None, out_file=None):
    """ This function downloads an audio file from a url to a file. """
    file_ = requests.get(url)
    pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    log.info(f"downloading {url} to {out_file}")
    with open(str(out_file), "wb") as f:
        f.write(file_.content)

        
class Timer():
    
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self.end = datetime.datetime.now()
        self.delta_t = self.end - self.start
        self.delta_ms = self.delta_t.total_seconds() * 1000
        log.info(f"timer : {self.name}, time elapsed (ms): {self.delta_ms}")
        