#!/usr/bin/env python3
# coding: utf-8
"""
    :brief: Script to download transcripts for a file
    given a file with YouTube urls
    :author: Shritishma Reddy
    :usage: `./transcript.py -i <inputfile> -o <outputfile> -p <proceesess>`

"""
import argparse
from itertools import chain
import json
from multiprocessing import Pool
from random import shuffle
from time import time

from pycaption import WebVTTReader
import pandas as pd
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
    :param df: whether to generate a dataframe (default: True)
    parser = argparse.ArgumentParser()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file",
                        type=str, required=True)
    parser.add_argument("-o", help="output file",
                        type=str, default="transcripts")
    parser.add_argument("-p", help="processes to use",
                        type=int, default=8)
    parser.add_argument("--df", "--dataframe", type=bool)
    args = parser.parse_args()
    return args


def processLink(link: str) -> str:
    """Process link to extract video subtitle

    :param link: str -> YouTube link
    :return tuple: unique_id, transcript -> The unique_id
        and video transcript
    """
    s_time = time()
    res = ydl.extract_info(link, download=False)
    unique_id, title = res['display_id'], res['title']
    print(f"{unique_id},{title}")


def process_transcript(trans: tuple) -> tuple:
    """Process the raw transcript object and create a list

    :param trans: raw captions
    :return transcript: list -> list of formatted caption strings
    """
    uid, trans = trans
    print(f"transcript: {uid}")
    transcript = [[uid, _.start, _.end, _.get_text()] for _ in trans]
    return transcript


def format_transcripts(args: argparse.Namespace, res: list, return_df: bool=False) -> dict:
    """Process the raw results and create a
    JSON given the transcript data

    :param res: tuple -> (uid, transcript) pair
    :param return_df: bool -> return the DataFrame generated
    :return final: dict -> JSON of results of the format
    """
    pool = Pool(args.p)
    resp = pool.map_async(process_transcript, res).get()
    pool.close()
    if return_df:
        columns = ['uid', 'start', 'end', 'text']
        resp = pd.DataFrame(chain(*resp), columns=columns)
        resp.to_csv(f"{args.o}.csv", index=False)
    else:
        json.dump(resp, open(f"{args.o}.json", 'a+'))
    return resp


def main():
    args = collect_args()
    links = open(args.i).read().splitlines()
    shuffle(links)  # to prevent predictable scrape pattern

    # fetch the links and transcripts
    pool = Pool(args.p)
    raw_res = pool.map_async(processLink, links).get()
    pool.close()

    # process raw transcripts downloaded
    results = format_transcripts(args, raw_res, return_df=args.df)
    print(results.head(10) if args.df else results[:10])


if __name__ == "__main__":
    main()

