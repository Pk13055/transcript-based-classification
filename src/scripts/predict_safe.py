#!/usr/bin/env python3
# coding: utf-8
"""
    :brief: Use this script to predict the safety of a video given
    its YouTube url.
    :usage: ./predict_safe.py -m /path/to/model -l <youtube url>
    :author: Pratik K
"""
import argparse
from io import BytesIO
import os
import pickle

from keras.models import load_model
import numpy as np
import pandas as pd
import pycaption
import requests
import youtube_dl


def collect_args() -> argparse.Namespace:
    """Parse command line arguments

    :return args: argparse.Namespace -> the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model to use to run the evaluation")
    parser.add_argument("-l", "--link", type=str, required=True,
                        help="YouTube link of the video")
    parser.add_argument("-t", "--tokenizer", type=str, help="Tokenizer path",
                        default=os.path.join(os.getcwd(), 'tokenizer.pkl'))
    args = parser.parse_args()
    return args


def parse_transcript(raw: pycaption.base.CaptionList) -> pd.DataFrame:
    """Given WebVTTReader caption list, parse
    to get a DataFrame

    :param raw: WebVTTReader caption
    :return df: DataFrame with cleaned caption data
    """
    captions = [[_.start * 1e-6, _.end * 1e-6, _.get_text()] for _ in raw]
    df = pd.DataFrame(captions, columns=['start', 'end', 'text'])
    return df


def get_transcript(link: str) -> pd.DataFrame:
    """Parse and collect transcript given link

    :param link: str -> YouTube uRL of the resource
    :return df: pd.DataFrame -> DataFrame containing required data
    """
    ydl = youtube_dl.YoutubeDL({
        'subtitlesformat': 'vtt',
        'quiet': True,
        'forcetitle': True,
        'writeautomaticsub': True,
        'simulate': True
    })
    raw = ydl.extract_info(link, download=False)
    unique_id, title = raw['display_id'], raw['title']
    print(f"Video - {unique_id}: {title}")
    try:
        sub_url = raw['requested_subtitles']['en']['url']
        resp = requests.get(sub_url, stream=True)
        bytes_ = BytesIO()
        [bytes_.write(block) for block in resp.iter_content(1024)]
        bytes_.seek(0)
        arr = pycaption.WebVTTReader().read(bytes_.read().decode('ascii'))
        transcript = arr.get_captions('en-US')
        df = parse_transcript(transcript)
    except KeyError:
        print(f"{title} [{unique_id}] has no English subtitles! Exiting ...")
        return pd.DataFrame([], columns=['text'])
    return df


def main():
    args = collect_args()

    print(f"Collecting transcript data for {args.link}")
    transcript = get_transcript(args.link)
    print(f"Transcript: {args.link}\n", transcript.head())
    X_test = transcript['text']

    print("\n\nProcessing data ...")
    tokenizer = pickle.load(open(args.tokenizer, 'rb'))
    encoded_ = tokenizer.texts_to_matrix(X_test)
    print(f"Data dimensions: {encoded_.shape}")

    print(f"\n\nLoading model {args.model.rsplit('/')[-1]}")

    model = load_model(args.model)
    print(model.summary())

    print("\n\nPredicting results ...")
    y_hat = model.predict([encoded_])
    print(y_hat)


if __name__ == "__main__":
    main()
