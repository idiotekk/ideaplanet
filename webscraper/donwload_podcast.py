import requests, os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import logging as log
from argparse import ArgumentParser, RawTextHelpFormatter
log.basicConfig(level=log.INFO)
# the 2 lines below makes browser invisible
chrome_options = Options()
chrome_options.add_argument("--headless")

def download_audio(url=None, out_file=None):
    """ Downloads an auodio file from a url to a file.
    """
    file_ = requests.get(url)
    log.info(f"downloading {url} to {out_file}")
    with open(out_file, "wb") as f:
        f.write(file_.content)

if __name__ == "__main__":

    parser = ArgumentParser(description="""This script downloads audio files from https://newbooksnetwork.com/
Example:
    donwload_podcast.py https://newbooksnetwork.com/emilie-hafner-burton-improving-human-rights-open-agenda-2021 """, formatter_class=RawTextHelpFormatter)
    parser.add_argument("url", type=str, help="url of the page")
    parser.add_argument("--out-dir", type=str, help="output folder of the audio", default=os.path.expandvars(f"/Users/$USER/Downloads/"))
    args = parser.parse_args()

    with webdriver.Chrome(options=chrome_options) as driver:

        driver.get(args.url)
        elem = driver.find_element_by_tag_name("iframe")
        player_url = elem.get_attribute("src")

        log.info(f"redirecting to {player_url}")
        driver.get(player_url)
        for elem in driver.find_elements_by_class_name("MenuBar__btn"):
            href = elem.get_attribute("href")
            if href and "mp3" in href:
                audio_url = href

    file_name = args.url.strip("/").split("/")[-1]
    download_audio(url=audio_url, out_file=(f"{args.out_dir}/{file_name}.mp3"))

