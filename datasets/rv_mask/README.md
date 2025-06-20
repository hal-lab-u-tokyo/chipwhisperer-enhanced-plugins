# RV-MASK

This dataset is obtained from the Boolean masking AES-128 implementation on a 32-bit RISC-V processor.

**Caution**: Extracting the full size version requires about 88 GB of disk space.
Therefore, we also provide a small size version of the dataset, which is a subset of the full dataset.

## Dataset features
- **Target**:  32-bit RISC-V processor with AES-128 implementation
  - **Board**: CW305
  - **Hardware Design**: [VexRiscv_SCA](https://github.com/hal-lab-u-tokyo/VexRiscv_SCA)
  - **Python Target Class**: `CW305RISCVAES128bit` (See Also [API Usage example](../../docs/hardware.md#aes-example-on-vexriscv_sca))
  - **Masking Scheme**: Boolean masking
  - **Software code**: [aes_soft](../../lib//cw_plugins/targets/aes_soft)
  - **C Compiler**: clang 19.1.7
  - **Operational frequency**: 5.0 MHz
- **Capture device**: ChipWhisperer-Lite
  - **Sampling rate**: 100 MS/s
- **Key**: Fixed single key (`2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c`)
- **Full size**
  - **Number of Traces**: 1,000,000 traces
  - **Number of Samples**: 11800 samples per trace (only for 1st round of AES)
-- **Small size**
  - **Number of Traces**: 150,000 traces
  - **Number of Samples**: 3000 samples per trace (\[3000:6000\] samples extracted from the full dataset)

## Setup dataset
We offer Makefile to setup the dataset, automatically downloading and extracting the dataset.
To set up the dataset, run the following command in this directory:
```bash
# for full size dataset
make rv_mask [PART_SIZE=NUM (=<100)]
# for small size dataset
make rv_mask_small [PART_SIZE=NUM (=<15)]
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


## Known successful attack
- **2nd-order CPA**
  - **Number of traces**: 150,000
  - **Point range**: `[3000:6000]`
  - **Leakage model**: Hamming weight of S-box output `cwa.leakage_models.sbox_output`
  - **Window size**: 1000

### Example of attack result
The reported sample positions are relative to the 3000th sample in the waveform. For example, a sample position of 1047 corresponds to the 4047th sample in the full waveform.
```
      Subkey  KGuess  SamplePos       Correlation     PGE
      00      0x2B    (1047,1806)     0.02373         0
      01      0x7E    (1467,2086)     0.02618         0
      02      0x15    (1328,1946)     0.02419         0
      03      0x16    (1026,1847)     0.01552         0
      04      0x28    (1187,1806)     0.01947         0
      05      0xAE    (1026,1807)     0.03112         0
      06      0xD2    (1467,2088)     0.01870         0
      07      0xA6    (1327,1946)     0.01622         0
      08      0xAB    (1327,1947)     0.02333         0
      09      0xF7    ( 987,1947)     0.02588         0
      10      0x15    (1026,1807)     0.02652         0
      11      0x88    (1468,2226)     0.01673         0
      12      0x09    (1466,2226)     0.02052         0
      13      0xCF    (1327,1946)     0.02157         0
      14      0x4F    (1187,1808)     0.01939         0
      15      0x3C    (1026,1947)     0.02269         0
```