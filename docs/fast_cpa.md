
# 1st-order CPA acceleration library

Our CPA acceleration library can be used with ChipWhisperer APIs to speed up the CPA process. It is designed to work with the ChipWhisperer Analyzer and can be used to analyze traces more efficiently.

Without our library, the CPA can be processed as the following code snippet.
However, this can be slow for large datasets (i.e., many traces or many points per trace)
since it is fully implemented in Python and cannot leverage multi-threading or GPU acceleration.

```python
import chipwhisper as cw
import chipwhisperer.analyzer as cwa
from chipwhisperer.analyzer.attacks import cpa_algorithms

# Open a ChipWhisperer project
proj = cw.open_project("path/to/your/project.cwp")

# select hypothetical leakage model
model = cwa.leakage_models.sbox_output
# make an attack instance using the selected model
attack = cwa.cpa(proj, model, algorithm=cpa_algorithms.Progressive)
# run the attack
results = attack.run(update_interval=1000)

# show the results
print(results)
```

## Provided Algorithms

This repository includes several CPA algorithms compatible with the CPA routine described above:

| Algorithm                     | Prerequisites                  | Remarks                                                                                                                                                                                                 |
|-------------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `FastCPAProgressive`          | OpenMP                         | Accelerates the CPA process using multithreading if OpenMP is available.                                                                                                                               |
| `FastCPAProgressiveTiling`    | OpenMP                         | Similar to `FastCPAProgressive`, but employs tiling to improve cache efficiency. Recommended for CPUs with sixteen or more cores for potentially higher performance.                                     |
| `FastCPAProgressiveCuda`      | CUDA, double-precision support | Leverages CUDA to accelerate the CPA process on NVIDIA GPUs. Requires native support for double-precision floating-point arithmetic.                                                                   |
| `FastCPAProgressiveCudaFP32`  | CUDA                           | A variant of `FastCPAProgressiveCuda` that uses single-precision floating-point arithmetic to emulate double-precision. Optimized for GPUs where single-precision performance significantly outpaces double-precision, such as GeForce series GPUs. |
| `FastCPAProgressiveOpenCL`    | OpenCL, double-precision support    | Uses OpenCL to accelerate the CPA process on GPUs. Designed for compatibility with a wide range of devices, including AMD GPUs.                                                                         |
| `FastCPAProgressiveOpenCLFP32`| OpenCL                         | A variant of `FastCPAProgressiveOpenCL` that uses single-precision floating-point arithmetic to emulate double-precision. Enables acceleration on FP32-only GPUs, such as Intel and Apple GPUs.          |

### Usage Example
As explained above, you can use the CPA acceleration library with the ChipWhisperer Analyzer API. So, just replace the `algorithm` parameter in the `cwa.cpa` function with one of the algorithms provided above.
For example, to use the `FastCPAProgressive` algorithm, you can do the following:
```python
from cw_plugins.analyzer.attacks.cpa_algorithms.fast_progressive import FastCPAProgressive
... # similar code as above
attack = cwa.cpa(proj, model, algorithm=FastCPAProgressive)
# run the attack
results = attack.run(update_interval=1000)
```
The other algorithms can be imported in the same way.

### Precision of Floating-point Arithmetic
To keep the compatibility with the ChipWhisperer's implementation, which uses `longdouble` type for correlation coefficient calculation,
we also use such an extended precision floating-point type.
However, we limit the precision-sensitive operations to the final correlation coefficient calculation.
However, some CPU architectures, such as Apple Silicon CPUs, do not support the extended precision floating-point type.
In such cases, we utilize a quad-precision floating-point emulation technique to ensure compatibility.
It is automatically applied when building the library on such architectures with cmake.

### OpenCL device selection
If you have multiple OpenCL devices, you can specify the device to use by setting the `CL_PLATFORM` and `CL_DEVICE` environment variables.

### Known Issues
The OpenCL implementation of FastCPA in Apple Silicon GPUs may provide incorrect results when a large number of traces and a long waveform are used.
In this case, the update interval should be set to a smaller value.

## Appropriate interval setting
All the algorithms described above utilize a progressive formulation, allowing CPA results to be computed incrementally using partial traces. This approach enables results to be updated repeatedly as more traces are processed. 

The `update_interval` parameter controls the number of traces processed in each update. While a larger `update_interval` can reduce the frequency of updates, it also increases memory usage. On GPUs with limited memory, setting a large `update_interval` may cause the algorithm to fail due to insufficient memory.

To ensure compatibility with GPUs that have constrained memory resources, it is recommended to use a smaller `update_interval`. This adjustment helps balance memory usage and performance, ensuring the algorithm runs smoothly.

## Supported Leakage Models
Among the available leakage models in the ChipWhisperer in `chipwhisperer.analyzer.attacks.models.AES128_8bit`,
compatible models for the following models are implemented in the CPA acceleration library:
- `cwa.leakage_models.plaintext_key_xor` (`PtKey_XOR`)
- `cwa.leakage_models.sbox_output` (`SBox_output`)
- `cwa.leakage_models.sbox_in_out_diff` (`SBoxInOutDiff`)
- `cwa.leakage_models.last_round_state` (`LastroundHW`)
- `cwa.leakage_models.last_round_state_diff` (`LastroundStateDiff`)
- `cwa.leakage_models.last_round_state_diff_alternate` (`LastroundStateDiffAlternate`)

## Instruction to add custom algorithms in the library
Currently, some source codes need to be modified to add new leakage models.

### Step 1: Create a new python implementation of the model

In your Python code, you need to create a new class derived from `AESLeakageHelper`.


```python
from chipwhisperer.analyzer.attacks.models.AES128_8bit import *
from chipwhisperer.analyzer.attacks.models.AES128_8bit import AESLeakageHelper
class CustomModel(AESLeakageHelper):
    def __init__(self):
        self.last_state = [0]*16
    def leakage(self, pt, ct, key, bnum):
		return leakage_value
# make it a compatible format to cwa.cpa method
custom_model = AES128_8bit(CustomModel)
```

### Step 2: Add corresponding C++ implementation
`cpp_libs/model/AESLeakageModel.cpp` is the file where the C++ implementation of the leakage model.
`cpp_libs/include/AESLeakageModel.h` is the header file for the C++ implementation.

First, you need to add the class definition in the header file.
```cpp
class CustomModel : public ModelBase
{
public:
	CustomModel() : ModelBase() {};
	~CustomModel() {};

	int leakage_impl(uint8_t * plaintext, uint8_t * ciphertext, uint8_t * key, int byte_index);
};
```
Then, you need to implement the `leakage_impl` method in the source file.
```cpp
int CustomModel::leakage_impl(uint8_t * plaintext, uint8_t * ciphertext, uint8_t * key, int byte_index)
{
	// Implement your custom leakage model logic here
	return leakage_value;
}
```

### Step 3: Binding the C++ implementation to Python
To bind the C++ implementation to Python, we employ `pybind11`. You need to append the binding code in `cpp_libs/model/bind.cpp` to support newly created model.

```cpp
PYBIND11_MODULE(model_kernel, module) {
// ...
py::class_<AESLeakageModel::CustomModel, AESLeakageModel::ModelBase>(module, "CustomModel")
		.def(py::init<>());
}
```

You need to modify the `lib/cw_plugins/analyzer/attacks/cpa_algorithms/models.py` file.
It defines `model_dict` which maps the python class to the C++ implementation.
So, just add the following line to the `model_dict` dictionary:
```python
model_dict = {
	# ...
	CustomModel: model_kernel.CustomModel,
}
```

# 2nd-order CPA and its acceleration library
In addition to the 1st-order CPA acceleration library, this repository also provides a 2nd-order CPA attack and its acceleration library.

The following code snippet shows how to use the 2nd-order CPA attack.

```python
from cw_plugins.analyzer.attacks.socpa import SOCPA, SOCPAAlgorithm

# Open a ChipWhisperer project
proj = cw.open_project("path/to/your/project.cwp")

# select hypothetical leakage model
model = cwa.leakage_models.sbox_output

# make an attack instance using the selected model and algorithm
attack = SOCPA(proj, model, SOCPAAlgorithm )

# optional setting example (not required for basic usage)
attack.window_size = 100 # window size of combined sample points
attack.set_trace_range(0, 1000) # limit the trace range to the first 1000 points
attack.point_range = (0, 500) # limit the point range to the first 500 points
# for tiling setting (only for SOCPAAlgorithm)
attack.set_trace_tile_size(32)
attack.set_point_tile_size(32)

# run the attack
results = attack.run()
```

The 2nd-order CPA attack supports the same models as the 1st-order CPA acceleration library. However, due to the increased need to preserve intermediate results in progressive formulations, our implementation of the 2nd-order CPA attack computes correlation coefficients for entire traces and points in a single step, bypassing the progressive formulation approach.
Therefore, the `update_interval` parameter is not applicable to the 2nd-order CPA attack.

## Supported Algorithms
The 2nd-order CPA acceleration library includes the following algorithms:
| Algorithm  | Prerequisites  | Remarks |
|------------|----------------|---------|
| `SOCPAAlgorithm` | (OpenMP) | Accelerates the 2nd-order CPA process using multithreading if OpenMP is available. To increase cache efficiency, it uses tiling to process traces and points in chunks. |
| `SOCPAAlgorithmCuda` | CUDA, double-precision support, 64bit atomicAdd | Leverages CUDA to accelerate the 2nd-order CPA process on NVIDIA GPUs. Optimized to utilize shared memory for improved performance.|
| `SOCPAAlgorithmCudaFP32` | CUDA, 64bit atomicAdd | A variant of `SOCPAAlgorithmCuda` that uses single-precision floating-point arithmetic to emulate double-precision. Optimized for GPUs where single-precision performance significantly outpaces double-precision, such as GeForce series GPUs. Also optimized to utilize shared memory for improved performance. |
| `SOCPAAlgorithmCudaNoSM` | CUDA, double-precision support | A variant of `SOCPAAlgorithmCuda` that does not use shared memory. |
| `SOCPAAlgorithmCudaFP32NoSM` | CUDA | A variant of `SOCPAAlgorithmCudaFP32` that does not use shared memory. |
| `SOCPAAlgorithmOpenCL` | OpenCL, double-precision support, 64-bit atom_cmpxchg | Uses OpenCL to accelerate the 2nd-order CPA process on GPUs. Optimized to utilize local memory for improved performance. Designed for compatibility with a wide range of devices, including AMD GPUs. |
| `SOCPAAlgorithmOpenCLFP32` | OpenCL, 64-bit atom_cmpxchg | A variant of `SOCPAAlgorithmOpenCL` that uses single-precision floating-point arithmetic to emulate double-precision. Optimized to utilize local memory for improved performance. |
| `SOCPAAlgorithmOpenCLNoSM` | OpenCL, double-precision support | A variant of `SOCPAAlgorithmOpenCL` that does not use local memory. |
| `SOCPAAlgorithmOpenCLFP32NoSM` | OpenCL | A variant of `SOCPAAlgorithmOpenCLFP32` that does not use local memory. |

Apple Silicon GPUs lack support for 64-bit atomic operations, even for integer types. As a result, the only compatible algorithm for these GPUs is `SOCPAAlgorithmOpenCLFP32NoSM`.

While Intel GPUs do support 64-bit atomic operations, their shared memory for all invoked threads is statically allocated within the kernel. This limitation generally results in insufficient shared memory, making `SOCPAAlgorithmOpenCLFP32NoSM` the sole viable algorithm for Intel GPUs as well.
