import os
import requests
from selenium import webdriver
# the 2 lines below are for printing logging messages 
import logging as log
log.basicConfig(level=log.INFO)
# the 3 lines below makes browser invisible
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")


def download_audio(url=None, out_file=None):
    """ Downloads an auodio file from a url to a file. """
    file_ = requests.get(url)
    log.info(f"downloading {url} to {out_file}")
    with open(out_file, "wb") as f:
        f.write(file_.content)


if __name__ == "__main__":

    # parse argument from commandline
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description="""This script downloads audio files from https://newbooksnetwork.com/
Example:
    run: python3 donwload_podcast.py 
    then type: https://newbooksnetwork.com/emilie-hafner-burton-improving-human-rights-open-agenda-2021 """, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--url", type=str, help="url of the page")
    parser.add_argument("--out-dir", type=str, help="output folder of the audio", default=os.path.expandvars(f"/Users/$USER/Downloads/"))
    args = parser.parse_args()
    
    # this line asks user to input the webpage address
    if args.url is None: args.url = input("input the webpage address here:")

    with webdriver.Chrome(options=chrome_options) as driver: # open a browser 

        # open the page and find the link to embedded player
        log.info(f"opening webpage: {args.url}")
        driver.get(args.url)
        elem = driver.find_element_by_tag_name("iframe")
        player_url = elem.get_attribute("src")

        # open the embedded player and find the link to the mp3
        log.info(f"redirecting to: {player_url}")
        driver.get(player_url)
        for elem in driver.find_elements_by_class_name("MenuBar__btn"):
            href = elem.get_attribute("href")
            if href and "mp3" in href:
                audio_url = href # this is the link to the mp3

    # save mp3 to your pc
    file_name = args.url.strip("/").split("/")[-1]
    download_audio(url=audio_url, out_file=(f"{args.out_dir}/{file_name}.mp3"))
