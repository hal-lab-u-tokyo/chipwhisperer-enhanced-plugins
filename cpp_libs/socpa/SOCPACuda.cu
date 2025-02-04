/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPACuda.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  01-02-2025 09:16:59
*    Last Modified: 01-02-2025 09:17:38
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "SOCPACuda.hpp"

namespace py = pybind11;

// check availability of OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda.h>

SOCPACuda::SOCPACuda(int num_traces, int num_points, int window_size, AESLeakageModel::ModelBase *model) :
	SOCPA(num_traces, num_points, window_size, model)
{

		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis,
					sizeof(int64_t) * byte_length * NUM_GUESSES));
		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis_square,
					sizeof(int64_t) * byte_length * NUM_GUESSES));

		// copy init data
		CUDA_CHECK(cudaMemcpy(device_sum_hypothesis, sum_hypothesis->get_pointer(),
								sum_hypothesis->get_size(),
								cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(device_sum_hypothesis_square,
								sum_hypothesis_square->get_pointer(), \
								sum_hypothesis_square->get_size(),
								cudaMemcpyHostToDevice));


		// init as nullptr
		// allocated at the first call of setup_arrays becasue the size is not known at this point
		device_hypothetial_leakage = nullptr;
		device_traces = nullptr;
		device_sum_hypothesis_trace = nullptr;
		device_sum_hypothesis_combined_trace = nullptr;

};

__global__
void sum_hypothesis_kernel(int byte_length, int num_guess, int num_traces,
	int *hypothetial_leakage, int64_t *sum_hypothesis, int64_t *sum_hypothesis_square)
{
	int byte_index = blockIdx.x;
	int guess = blockIdx.y * blockDim.x + threadIdx.x;
	if (guess < num_guess) {
		int64_t sum_hyp = 0;
		int64_t sum_hyp_square = 0;
		for (int t = 0; t < num_traces; t++) {
			int hyp = hypothetial_leakage[byte_index * num_guess * num_traces + guess * num_traces + t];
			sum_hyp += hyp;
			sum_hyp_square += hyp * hyp;
		}
		sum_hypothesis[byte_index * num_guess + guess] += sum_hyp;
		sum_hypothesis_square[byte_index * num_guess + guess] += sum_hyp_square;
	}
}

__global__
void sum_hypothesis_trace_kernel(int byte_length, int num_guess,
	int num_traces, int num_points, int *hypothetial_leakage, double *traces,
	double *sum_hypothesis_trace)
{
	int byte_index = blockIdx.x;
	int guess = blockIdx.y * blockDim.x + threadIdx.x;
	int point_index = blockIdx.z * blockDim.y + threadIdx.y;

	if (guess < num_guess && point_index < num_points) {
		double sum = 0;
		for (int trace_index = 0; trace_index < num_traces; trace_index++) {
			int hyp = hypothetial_leakage[byte_index * num_guess * num_traces + guess * num_traces + trace_index];
			sum += hyp * traces[trace_index * num_points + point_index];
		}
		sum_hypothesis_trace[byte_index * num_guess * num_points + guess * num_points + point_index] += sum;
	}

}


void SOCPACuda::update_sum_hypothesis_trace() {
	// updata sum_hypothesis, sum_hypothesis_square
	dim3 dimBlock(32);
	dim3 dimGrid(byte_length, (NUM_GUESSES + dimBlock.x - 1) / dimBlock.x);
	sum_hypothesis_kernel<<<dimGrid, dimBlock>>>(byte_length, NUM_GUESSES, num_traces,
		device_hypothetial_leakage, device_sum_hypothesis, device_sum_hypothesis_square);
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}
	// offload to GPU for sum_hypothesis_trace
	dimBlock = dim3(32, 32);
	dimGrid = dim3(byte_length, (NUM_GUESSES + dimBlock.x - 1) / dimBlock.x, (num_points + dimBlock.y - 1) / dimBlock.y);
	sum_hypothesis_trace_kernel<<<dimGrid, dimBlock>>>(byte_length, NUM_GUESSES, num_traces, num_points,
		device_hypothetial_leakage, device_traces, device_sum_hypothesis_trace);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}

	// copy back
	CUDA_CHECK(cudaMemcpy((int64_t*)sum_hypothesis->get_pointer(),
							device_sum_hypothesis,
							sum_hypothesis->get_size(),
							cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((int64_t*)sum_hypothesis_square->get_pointer(),
							device_sum_hypothesis_square,
							sum_hypothesis_square->get_size(),
							cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((double*)sum_hypothesis_trace->get_pointer(),
							device_sum_hypothesis_trace,
							sum_hypothesis_trace->get_size(),
							cudaMemcpyDeviceToHost));
}


const int threadPerBlock = (4 * 16 * 8);

__global__
void sum_hypothesis_coumbined_trace_kernel(
	int num_traces, int num_points, int window_size,
	int *hypothetial_leakage, double *traces,
	double *sum_hypothesis_combined_trace)
{
	__shared__ double cache[threadPerBlock];

	int point_index = threadIdx.x + blockIdx.x * blockDim.x;
	int window_index = threadIdx.y + blockIdx.y * blockDim.y;

	int cache_offset = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z;
	int cache_index = threadIdx.z + cache_offset;
	int trace_index = threadIdx.z;

	if (point_index + window_index < num_points - 1 && window_index < window_size) {
		cache[cache_index] = 0;
		while (trace_index < num_traces) {
			int hyp = hypothetial_leakage[trace_index];
			cache[cache_index] += hyp * traces[trace_index * num_points + point_index + window_index + 1] *
			traces[trace_index * num_points + point_index];
			trace_index += blockDim.z;
		}
	}
	__syncthreads();

	// reduction
	if (threadIdx.z == 0 &&  window_index < window_size) {
		double sum = 0;
		for (int i = 0; i < blockDim.z; i++) {
			sum += cache[i + cache_offset];
		}
		sum_hypothesis_combined_trace[point_index * window_size + window_index] = sum;
	}

}

void SOCPACuda::update_sum_hypothesis_combined_trace()
{

	dim3 dimBlock(4, 16, 8);
	dim3 dimGrid((num_points + dimBlock.x - 1) / dimBlock.x,
				(window_size + dimBlock.y - 1) / dimBlock.y, 1);

	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int guess = 0; guess < NUM_GUESSES; guess++) {
			sum_hypothesis_coumbined_trace_kernel<<<dimGrid, dimBlock>>>(num_traces, num_points, window_size,
				&device_hypothetial_leakage[byte_index * NUM_GUESSES * num_traces + guess * num_traces], device_traces,
				device_sum_hypothesis_combined_trace);
			auto err = cudaGetLastError();
			if (err != cudaSuccess) {
				throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
			}
			// copy back
			double* dst = &(((double*)sum_hypothesis_combined_trace->get_pointer())[byte_index * NUM_GUESSES * num_points * window_size + guess * num_points * window_size]);
			CUDA_CHECK(cudaMemcpy(dst, device_sum_hypothesis_combined_trace,
									sizeof(double) * num_points * window_size,
									cudaMemcpyDeviceToHost));
		}
	}

}

void SOCPACuda::setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey) {
	SOCPA::setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);

	// malloc gpu memory
	if (device_hypothetial_leakage == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_hypothetial_leakage,
						hypothetial_leakage->get_size()));
	}
	if (device_sum_hypothesis_trace == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis_trace,
						sum_hypothesis_trace->get_size()));
	}

	if (device_sum_hypothesis_combined_trace == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis_combined_trace,
						sizeof(double) * num_points * window_size));
	}

	if (device_traces == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_traces,
			traces->get_size()));
	}
	// copy trace data
	CUDA_CHECK(cudaMemcpy(device_traces, traces->get_pointer(),
			traces->get_size(),
			cudaMemcpyHostToDevice));
}

void SOCPACuda::calculate_hypothesis()
{
	SOCPA::calculate_hypothesis();
	// copy CPU calculated hypothetial_leakage to GPU
	CUDA_CHECK(cudaMemcpy(device_hypothetial_leakage,
							hypothetial_leakage->get_pointer(),
							hypothetial_leakage->get_size(),
							cudaMemcpyHostToDevice));
}

SOCPACuda::~SOCPACuda() {
	cudaFree(device_traces);
	cudaFree(device_hypothetial_leakage);
	cudaFree(device_sum_hypothesis);
	cudaFree(device_sum_hypothesis_square);
	cudaFree(device_sum_hypothesis_trace);
	cudaFree(device_sum_hypothesis_combined_trace);

};

PYBIND11_MODULE(socpa_cuda_kernel, module) {
	module.doc() = "CUDA implemetation plugin for SOCPA";

	py::class_<SOCPACuda,SOCPA>(module, "SOCPACuda")
		.def(py::init<int, int, int, AESLeakageModel::ModelBase*>());

}

