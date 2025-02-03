/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPACuda.cu
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:57:31
*    Last Modified: 17-02-2024 22:08:56
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "FastCPACuda.hpp"

namespace py = pybind11;

// check availability of OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda.h>


FastCPACuda::FastCPACuda(int num_traces, int num_points, AESLeakageModel::ModelBase *model) : FastCPA(num_traces, num_points, model) {

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

};

FastCPACuda::~FastCPACuda() {
	cudaFree(device_traces);
	cudaFree(device_hypothetial_leakage);
	cudaFree(device_sum_hypothesis);
	cudaFree(device_sum_hypothesis_square);
	cudaFree(device_sum_hypothesis_trace);

};

void FastCPACuda::setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey) {
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
	// copy trace data
	CUDA_CHECK(cudaMemcpy(device_traces, traces->get_pointer(),
			traces->get_size(),
			cudaMemcpyHostToDevice));
}

void FastCPACuda::calculate_hypothesis()
{
	FastCPA::calculate_hypothesis();
	// copy CPU calculated hypothetial_leakage to GPU
	CUDA_CHECK(cudaMemcpy(device_hypothetial_leakage,
							hypothetial_leakage->get_pointer(),
							hypothetial_leakage->get_size(),
							cudaMemcpyHostToDevice));
}

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


void FastCPACuda::calculate_correlation_subkey(Array3D<TRACE_T>* diff, QUADFLOAT *sumden2) {

	// offload to GPU for sum_hypothesis, sum_hypothesis_square
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


	#ifdef _OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int guess = 0; guess < NUM_GUESSES; guess++) {
			// calc sumden1
			QUADFLOAT sumden1 = SQUARE(sum_hypothesis->at(byte_index, guess))
				- total_traces * sum_hypothesis_square->at(byte_index, guess);
			// calc sumnum
			QUADFLOAT sumnum;
			for (int p = 0; p < num_points; p++) {
				sumnum =
					(QUADFLOAT)total_traces * sum_hypothesis_trace->at(byte_index, guess, p)
					- sum_trace[p] * sum_hypothesis->at(byte_index, guess);
				diff->at(byte_index, guess, p) = (TRACE_T)sumnum / std::sqrt((TRACE_T)sumden1 * (TRACE_T)sumden2[p]);
			}
		}
	}
}

PYBIND11_MODULE(cpa_cuda_kernel, module) {
	module.doc() = "CUDA implemetation plugin for CPA";

	py::class_<FastCPACuda,FastCPA>(module, "FastCPACuda")
		.def(py::init<int, int, AESLeakageModel::ModelBase*>());

}


