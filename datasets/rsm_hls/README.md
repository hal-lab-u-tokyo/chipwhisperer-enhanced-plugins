# RSM-HLS Dataset

This dataset is obtained from HLS implementation of RSM masking scheme.

## Dataset features
- **Target**: HLS implementation of AES-128 with RSM masking scheme
  - **Board**: CW305
  - **Hardware Design**: aes128_rsm_hls in [sca_design_repo](https://github.com/hal-lab-u-tokyo/sca_design_repo)
  - **Python Target Class**: `CW305ShellExampleAES128BitHLS` (See Also [API Usage example](../../docs/hardware.md#example-implementations-of-cw305-shell))
  - **HLS tool**: AMD Vitis HLS 2023.2
  - **HLS target frequency**: 20 MHz (1 cycle per AES round)
  - **Operational frequency**: 5.0 MHz
  - **Seed for mask table creation**: `10` (default value)
- **Capture device**: ChipWhisperer-Lite
  - **Sampling rate**: 100 MS/s
- **Key**: Fixed single key (`2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c`)
- **Size**
  - **Number of Traces**: 1,000,000 traces
  - **Number of Samples**: 500 samples per trace

## Setup dataset
We offer Makefile to setup the dataset, automatically downloading and extracting the dataset.
To set up the dataset, run the following command in this directory:
```bash
make rsm_hls [PART_SIZE=NUM (=<100)]
```

ChipWhisperer projects split the trace data into multiple parts, with each part containing 10,000 traces. You can control how many parts to extract by setting the `PART_SIZE` variable in the command line. For instance, specifying `PART_SIZE=5` will extract the first 50,000 traces from the dataset.

If you need to remove intermediate files, use the following command:
```bash
make clean
```

To revert the dataset to its original state, use:
```bash
make distclean
```

