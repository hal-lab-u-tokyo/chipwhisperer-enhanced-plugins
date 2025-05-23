/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPACudaFP32.cu
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-05-2025 17:18:43
*    Last Modified: 23-05-2025 19:01:16
*/

#include "FastCPACudaFP32.hpp"

#include <fp32_premitives.hpp>

namespace py = pybind11;

// check availability of OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda.h>
#include <cuda_profiler_api.h>

inline float2 to_float2(double x)
{
	const double SPLITTER = (1u << 29u) + 1u;
	double t = SPLITTER * x;
	double t_hi = t - (t - x);
	double t_lo = x - t_hi;
	return float2({(float)t_hi, (float)t_lo});
}

inline double to_double(float2 x)
{
	return (double)x.x + (double)x.y;
}


FastCPACudaFP32::~FastCPACudaFP32()
{
	if (traces_df64 != nullptr)
		delete traces_df64;
	if (sum_hypothesis_trace_df64 != nullptr)
		delete sum_hypothesis_trace_df64;
}

__global__
void sum_hypothesis_trace_kernel(int byte_length, int num_guess,
	int num_traces, int num_points, int *hypothetial_leakage, float2 *traces,
	float2 *sum_hypothesis_trace)
{
	int byte_index = blockIdx.x;
	int guess = blockIdx.y * blockDim.x + threadIdx.x;
	int point_index = blockIdx.z * blockDim.y + threadIdx.y;

	if (guess < num_guess && point_index < num_points) {
		// float2 sum = sum_hypothesis_trace[byte_index * num_guess * num_points + guess * num_points + point_index];
		int index = byte_index * num_guess * num_points + guess * num_points + point_index;
		float2 sum = {0.0f, 0.0f};
		for (int trace_index = 0; trace_index < num_traces; trace_index++) {
			float hyp = (float)hypothetial_leakage[byte_index * num_guess * num_traces + guess * num_traces + trace_index];
			float2 trace = traces[trace_index * num_points + point_index];
			sum = df64_add(sum, df64_mul({hyp, 0.0f}, trace));
		}
		sum_hypothesis_trace[index] = df64_add(sum_hypothesis_trace[index], sum);
	}
}

void FastCPACudaFP32::run_sum_hypothesis_trace_kernel() {
	// offload to GPU for sum_hypothesis_trace
	dim3 dimBlock = dim3(8, 32);
	dim3 dimGrid = dim3(byte_length, (NUM_GUESSES + dimBlock.x - 1) / dimBlock.x, (num_points + dimBlock.y - 1) / dimBlock.y);

	sum_hypothesis_trace_kernel<<<dimGrid, dimBlock>>>(byte_length, NUM_GUESSES, num_traces, num_points,
		device_hypothetial_leakage, (float2*)device_traces, (float2*)device_sum_hypothesis_trace);
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}

	// copy back
	CUDA_CHECK(cudaMemcpy((float2*)sum_hypothesis_trace_df64->get_pointer(),
							device_sum_hypothesis_trace,
							sum_hypothesis_trace_df64->get_size(),
							cudaMemcpyDeviceToHost));
	// convert float2 to double
	float2_to_double((float2*)sum_hypothesis_trace_df64->get_pointer(), (RESULT_T*)sum_hypothesis_trace->get_pointer(), byte_length * NUM_GUESSES * num_points);
}


void FastCPACudaFP32::setup_arrays(py::array_t<TRACE_T> &py_traces,
	py::array_t<uint8_t> &py_plaintext,
	py::array_t<uint8_t> &py_ciphertext,
	py::array_t<uint8_t> &py_knownkey)
{

	FastCPA::setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);

	// malloc gpu memory
	if (device_hypothetial_leakage == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_hypothetial_leakage,
						hypothetial_leakage->get_size()));
	}
	if (device_sum_hypothesis_trace == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis_trace,
					sum_hypothesis_trace->get_size()));
		CUDA_CHECK(cudaMemcpy(device_sum_hypothesis_trace,
					sum_hypothesis_trace->get_pointer(),
					sum_hypothesis_trace->get_size(),
					cudaMemcpyHostToDevice));
	}

	if (device_traces == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_traces,
			traces->get_size()));
	}

	// allocate memory for format conversion
	if (traces_df64 == nullptr) {
		traces_df64 = new Array2D<float2>(num_traces, num_points);
	}
	if (sum_hypothesis_trace_df64 == nullptr) {
		sum_hypothesis_trace_df64 = new Array3D<float2>(byte_length, NUM_GUESSES, num_points);
	}

	// convert double to float2
	double_to_float2(traces->get_pointer(), (float2*)traces_df64->get_pointer(), num_traces * num_points);

	// copy trace data
	CUDA_CHECK(cudaMemcpy(device_traces, traces_df64->get_pointer(),
			traces_df64->get_size(),
			cudaMemcpyHostToDevice));

}

void FastCPACudaFP32::double_to_float2(const double *src, float2 *dst, int length) {

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_float2(src[i]);
	}
}

void FastCPACudaFP32::float2_to_double(const float2 *src, double *dst, int length) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_double(src[i]);
	}
}