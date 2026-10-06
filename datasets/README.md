## Datasets for Side-Channel Attacks
This repository offers trace datasets collected using this framework. These datasets are designed to facilitate benchmarking and the evaluation of side-channel attack techniques.
These datasets are specifically tailored for non-profiling side-channel attacks.
In this context, all traces correspond to the fixed single key,
and no masking values are included.

**Note that this repository does not include actual data files due to their large size. Follow the instructions in each dataset's README file to prepare the datasets.**

### List of Datasets

- [RV-MASK](./rv_mask/README.md): Boolean masking AES-128 on 32-bit RISC-V processor
- [RSM-RTL](./rsm_rtl/README.md): RTL implementation of RSM-AES
- [RSM-HLS](./rsm_hls/README.md): HLS implementation of RSM-AES
