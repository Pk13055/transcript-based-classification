#!/usr/bin/env python3
# coding: utf-8
"""
    :brief: Script to download transcripts for a file
    given a file with YouTube urls
    :author: Shritishma Reddy
    :usage: `./transcript.py -i <inputfile> -o <outputfile> -p <proceesess>`

"""
import argparse
from multiprocessing import Pool
from random import shuffle
from time import time

from pycaption import WebVTTReader
import requests
import youtube_dl


ydl_opts = {
    'subtitlesformat': 'vtt',
    'quiet':True,
    'forcetitle':'Force printing title.',
    'writeautomaticsub': True,
    'simulate': True
}
ydl = youtube_dl.YoutubeDL(ydl_opts)


def collect_args():
    """Command line args

    :param i: input file (links)
    :param o: output file (transcripts)
    :param p: number of processes to use (default: 8)
    parser = argparse.ArgumentParser()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file",
                        type=str, required=True)
    parser.add_argument("-o", help="output file",
                        type=str, default="transcripts.csv")
    parser.add_argument("-p", help="processes to use",
                        type=int, default=8)
    args = parser.parse_args()
    return args


def processLink(link: str) -> str:
    """Process link to extract video subtitle

    :param link: str -> YouTube link
    :return transcript: str -> The video transcript
    """
    s_time = time()
    res = ydl.extract_info(link, download=False)
    unique_id = res['display_id']
    transcript = ""
    if 'requested_subtitles' in res and 'en' in res['requested_subtitles']:
        url = res['requested_subtitles']['en']['url']
        resp = requests.get(url, stream=True)
        captions = WebVTTReader().read(resp.text)
        transcript = captions.get_captions('en-US')
    print(f"Processed '{res['title']}' | {time() - s_time}ms")
    return transcript, unique_id


def main():
    args = collect_args()
    links = open(args.i).read().splitlines()
    shuffle(links)  # to prevent predictable scrape pattern
    transcripts = [processLink(link) for link in links]
    pool = Pool(8)
    results = pool.map_async(processLink, links).get()
    pool.close()
    print(results)

if __name__ == "__main__":
    main()
