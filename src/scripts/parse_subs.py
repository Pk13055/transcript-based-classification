#!/usr/bin/env python3
# coding: utf-8
"""
    :author: Pratik K
    :brief: Converts an srt sub file to a csv
"""
import os
import glob
from datetime import datetime
import argparse
from multiprocessing import Pool
from itertools import chain
import json
import re

import pandas as pd
import numpy as np
pattern = re.compile(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]')


def collect_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str,
                        default='data/subs',
                        help="directory with the sub files")
    parser.add_argument('-t', '--threads', type=int,
                        default=8,
                        help="number of threads to use")
    args = parser.parse_args()
    return args


def get_files(_dir: str) -> list:
    """Get the srt files in a directory

    :param dir: str -> The path to work with
    :return files: list -> File list to scrape
    """
    return glob.glob(os.path.join(_dir, '*.srt'))


def convert(timestamps: list) -> float:
    res = []
    start, end = timestamps
    end = end.split(' ')[0] if 'line' in end else end
    for timestamp in [start, end]:
        t = datetime.strptime(timestamp, "%H:%M:%S,%f")
        res.append(t.hour * 60 * 60 + t.minute * 60 + t.second + t.microsecond * 1e-6)
    return res


def get_trans(idx, path: str) -> list:
    """Parse the raw text into a list based subtitle structure

    :param path: str -> The path to the file
    :return subs: list -> List of list represented:
        [[idx, id, start, end, text], ...]
    """
    raw = open(path).read().splitlines()
    raw = [_.strip(' ') for _ in raw if _ not in ['', ' ', '\n']]
    clean = lambda x: re.sub(pattern, '', x)
    idxs = [idx for idx, _ in enumerate(raw) if _.isdigit()] + [None]
    trans = [raw[start:stop] for start, stop in zip(idxs, idxs[1:])]
    trans = [[idx, int(_[0]),
              *convert(_[1].strip(' ').split(' --> ')),
              clean(''.join(_[2:]))] for _ in trans]
    return trans


def main():
    args = collect_args()
    files = get_files(args.dir)

    pool = Pool(args.threads)
    res = pool.starmap_async(get_trans, enumerate(files)).get()
    res.sort()
    raw = np.array(list(chain(*res)))
    df = pd.DataFrame(raw, columns=['subtitle', 'idx', 'start', 'end', 'text'])
    df['subtitle'] = df['subtitle'].map(lambda x: files[int(x)])
    df.to_csv('raw_subs.csv', index=False)
    print(df.head(10))

if __name__ == "__main__":
    main()

