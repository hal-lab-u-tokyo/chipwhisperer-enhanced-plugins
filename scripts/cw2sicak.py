#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
#   Copyright (C) 2025 The University of Tokyo
#   
#   File:          /scripts/cw2sicak.py
#   Project:       sca_toolbox
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  21-03-2025 17:54:02
#   Last Modified: 24-05-2025 06:50:14
###
#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import chipwhisperer as cw
import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

from chipwhisperer.analyzer.attacks.models.AES128_8bit import SBox_output, LastroundStateDiff

def arg_parse():
    parser = ArgumentParser(description="Utility script to convert ChipWhisperer project to SICAK compatible format.")
    parser.add_argument("project", help="Path to chipwhisperer project file.", type=str)
    parser.add_argument("output", help="Path to output directory.", type=str)
    parser.add_argument("--prefix", help="Prefix for output files", default=None)
    parser.add_argument("--leak_model", help='Leak model to use. Default is "sbox_output"', default="sbox_output", choices=["sbox_output", "last_round_state_diff"])
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

    if args.leak_model == "sbox_output":
        model = SBox_output()
    elif args.leak_model == "last_round_state_diff":
        model = LastroundStateDiff()
    else:
        print(f"Error: Unknown leak model '{args.leak_model}'")
        return

    prefix = args.prefix + "_" if args.prefix is not None else ""
    power_trace_file = os.path.join(args.output, f"{prefix}power_traces.bin")
    hypo_file = os.path.join(args.output, f"{prefix}hypo.bin")


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

    hypo_array = np.ndarray(shape=(16, num_traces, 256), dtype=np.uint8)


    # selected traces
    power_traces = np.array([t.wave[sample_range] for t in project.traces[:num_traces]])
    max_ampl = np.max(power_traces)
    min_ampl = np.min(power_traces)

    # normalized
    power_traces = (power_traces - min_ampl) / (max_ampl - min_ampl)
    # scale to signed 16-bit
    power_traces = (power_traces * 32767).astype(np.int16)

    pbar = tqdm(total=256*num_traces*16)
    for byte in range(16):
        key = [0]*16

        for guess in range(256):
            key[byte] = guess
            for i in range(num_traces):
                t = project.traces[i]
                hyp = model.leakage(t.textin, t.textout, key, byte)
                hypo_array[byte, i, guess] = hyp
                pbar.update(1)
    pbar.close()


    power_traces.tofile(power_trace_file)
    hypo_array.tofile(hypo_file)

if __name__ == "__main__":
    main()
