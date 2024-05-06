#!/usr/bin/env python3
# -*- coding:utf-8 -*-

###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /scripts/cw2ramdpa.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  06-05-2024 17:34:17
#   Last Modified: 06-05-2024 19:11:01
###


import chipwhisperer as cw

from argparse import ArgumentParser
import os
import numpy as np


def arg_parse():
    parser = ArgumentParser(description="Utility script to convert ChipWhisperer project to Daredevil compatible format.")
    parser.add_argument("project", help="Path to chipwhisperer project file.", type=str)
    parser.add_argument("output", help="Path to output directory.", type=str)
    parser.add_argument("--prefix", help="Prefix for output files", default=None)
    parser.add_argument("--num-trace", help="Number of traces to use. Default is all.", default=None, type=int)
    parser.add_argument("--point-range", help="Range of points to use. Default is all.", default=None, type=int, nargs=2)

    return parser.parse_args()


def main():
    args = arg_parse()
    try:
        project = cw.open_project(args.project)
    except Exception as e:
        print(f"Error: {e}")
        return

    # create output directory
    try:
        os.makedirs(args.output, exist_ok=False)
    except FileExistsError:
        print(f"Error: Output directory '{args.output}' already exists.")
        return

    # save config file
    if args.num_trace is None:
        num_traces = len(project.traces)
    else:
        num_traces = args.num_trace
    if args.point_range is None:
        num_samples = len(project.traces[0].wave)
        sample_range = slice(0, num_samples)
    else:
        num_samples = args.point_range[1] - args.point_range[0]
        sample_range = slice(args.point_range[0], args.point_range[1])

    traces_filename = os.path.join(args.output, \
                                   ("" if args.prefix is None else args.prefix + "_") + "trace.txt")
    plaintext_filename = os.path.join(args.output, \
                                   ("" if args.prefix is None else args.prefix + "_") + "plaintext.txt")

    max_ampl = np.max([np.max(t.wave[sample_range]) for t in project.traces])
    min_ampl = np.min([np.min(t.wave[sample_range]) for t in project.traces])
    trace_data = np.empty(shape=(num_traces, num_samples), dtype=np.uint32)
    for i, t in enumerate(project.traces[:num_traces]):
        wave = ((t.wave[sample_range] - min_ampl) / (max_ampl - min_ampl)) * 1024 + 83
        wave = wave.astype(np.uint32)
        trace_data[i][:] = wave

    plaintext_data = np.array(project.textins[:num_traces], dtype=np.uint8)

    np.savetxt(traces_filename, trace_data, fmt="%03.3d")
    np.savetxt(plaintext_filename, plaintext_data, fmt="%03.3d")

if __name__ == "__main__":
    main()