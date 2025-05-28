#!/usr/bin/env python3
# -*- coding:utf-8 -*-

###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /Daredevil/cw2daredevil.py
#   Project:       kojimatakuya
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  03-05-2024 17:15:24
#   Last Modified: 16-05-2024 17:40:05
###

import chipwhisperer as cw

from argparse import ArgumentParser
import os
import numpy as np
from multiprocessing import cpu_count


dtype_dict = {"float": np.float32, "double": np.float64, "int8": np.int8}
dtype_label_dict = {"float": "f", "double": "d", "int8": "i"}

def arg_parse():
    parser = ArgumentParser(description="Utility script to convert ChipWhisperer project to Daredevil compatible format.")
    parser.add_argument("project", help="Path to chipwhisperer project file.", type=str)
    parser.add_argument("output", help="Path to output directory.", type=str)
    parser.add_argument("--prefix", help="Prefix for output files", default=None)
    parser.add_argument("--data-type", help="Data type to export. Default is 'float'.", default="float", choices=dtype_dict.keys())
    parser.add_argument("--num-threads", help=f"Number of threads to use. Default is {cpu_count()}.", default=cpu_count(), type=int)
    parser.add_argument("--leak_model", help="Leak model to use. Default is 'AES_AFTER_SBOX'", default="AES_AFTER_SBOX", choices=["AES_AFTER_SBOX", "AES_BEFORE_SBOX", "AES_AFTER_SBOXINV"])
    parser.add_argument("--num-trace", help="Number of traces to use. Default is all.", default=None, type=int)
    parser.add_argument("--point-range", help="Range of points to use. Default is all.", default=None, type=int, nargs=2)

    return parser.parse_args()

CONFIG_FMT = """
[Traces]
files=1
trace_type={dtype_label}
transpose=true
index=0
nsamples={num_samples}
trace={traces_filename} {num_traces} {num_samples}

[Guesses]
files=1
guess_type=u
transpose=true
guess={plaintext_filename} {num_traces} {plaintext_len}

[General]
threads={num_threads}
order=1
# window=4
return_type=double
algorithm=AES
position=LUT/{model}
round=0
bitnum=none
bytenum=all
correct_key={hex_key}
memory=4G
top=20
"""



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

    config_filename = os.path.join(args.output, \
                                   ("" if args.prefix is None else args.prefix + "_") + "config")
    traces_filename = os.path.join(args.output, \
                                   ("" if args.prefix is None else args.prefix + "_") + "trace.bin")
    plaintext_filename = os.path.join(args.output, \
                                   ("" if args.prefix is None else args.prefix + "_") + "plaintext.bin")

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

    plaintext_len = len(project.textins[0])
    dtype_label = dtype_label_dict[args.data_type]
    hex_key = "0x"
    for kb in project.keys[0]:
        hex_key += f"{kb:02x}"

    config_str = CONFIG_FMT.format(dtype_label=dtype_label, num_samples=num_samples, \
                                   num_traces=num_traces, traces_filename=traces_filename, \
                                   plaintext_filename=plaintext_filename, \
                                   plaintext_len=plaintext_len, num_threads=args.num_threads, \
                                   model=args.leak_model, hex_key=hex_key)
    with open(config_filename, "w") as cf:
        cf.write(config_str)

    # selected traces
    traces = np.array([t.wave[sample_range] for t in project.traces[:num_traces]])
    max_ampl = np.max(traces)
    min_ampl = np.min(traces)
    # normalized
    traces = (traces - min_ampl) / (max_ampl - min_ampl)

    # save traces and plaintext
    with open(traces_filename, "wb") as tf, open(plaintext_filename, "wb") as pf:
        for i, t in enumerate(project.traces[:num_traces]):
            tf.write(np.array(traces[i]).astype(dtype_dict[args.data_type]).tobytes())
            pf.write(np.array(t.textin, dtype=np.uint8).tobytes())


if __name__ == "__main__":
    main()
