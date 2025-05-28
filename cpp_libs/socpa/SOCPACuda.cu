/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPACuda.cu
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  01-02-2025 09:16:59
*    Last Modified: 28-05-2025 02:37:38
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "SOCPACuda.hpp"
#include "SOCPACudaFP32.hpp"

namespace py = pybind11;

// check availability of OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <algorithm>


SOCPACuda::SOCPACuda(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model,
	bool use_shared_mem) :
	SOCPA(byte_length, num_points, window_size, model),
	use_shared_mem(use_shared_mem)
{
		// get device properties
		cudaDeviceProp prop;
		int device;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&prop, device);
		shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
		global_mem_capacity = prop.totalGlobalMem;

		// determine tile size no to ocupy more than 50% of the global memory for the temporary array
		point_tile_size = 1 << static_cast<int>(std::ceil(std::log2(num_points)));

		while ((sizeof(double) * point_tile_size * window_size) > (global_mem_capacity / 2)) {
			point_tile_size /= 2;
		}

		point_tile_size = std::min(point_tile_size, num_points);

		// allocate memory on the GPU
		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis,
					sizeof(int64_t) * byte_length * NUM_GUESSES));
		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis_square,
					sizeof(int64_t) * byte_length * NUM_GUESSES));

		CUDA_CHECK(cudaMalloc((void**)&device_sum_trace_x_win,
					sizeof(double) * num_points * window_size));
		CUDA_CHECK(cudaMalloc((void**)&device_sum_trace2_x_win,
					sizeof(double) * num_points * window_size));
		CUDA_CHECK(cudaMalloc((void**)&device_sum_trace_x_win2,
					sizeof(double) * num_points * window_size));
		CUDA_CHECK(cudaMalloc((void**)&device_sum_trace2_x_win2,
					sizeof(double) * num_points * window_size));


		CUDA_CHECK(cudaMalloc((void**)&device_sum_hypothesis_combined_trace,
						sizeof(double) * point_tile_size * window_size));

		sum_hypothesis_combined_trace = new double[point_tile_size * window_size];

		// init as nullptr
		// allocated at the first call of setup_arrays becasue the size is not known at this point
		device_hypothetial_leakage = nullptr;
		device_traces = nullptr;
		device_sum_hypothesis_trace = nullptr;

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
		sum_hypothesis[byte_index * num_guess + guess] = sum_hyp;
		sum_hypothesis_square[byte_index * num_guess + guess] = sum_hyp_square;
	}
}

void SOCPACuda::calculate_sum_hypothesis()
{
	// updata sum_hypothesis, sum_hypothesis_square
	dim3 dimBlock(32);
	dim3 dimGrid(byte_length, (NUM_GUESSES + dimBlock.x - 1) / dimBlock.x);
	sum_hypothesis_kernel<<<dimGrid, dimBlock>>>(byte_length, NUM_GUESSES, num_traces,
		device_hypothetial_leakage, device_sum_hypothesis, device_sum_hypothesis_square);
	auto err = cudaGetLastError();
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
		sum_hypothesis_trace[byte_index * num_guess * num_points + guess * num_points + point_index] = sum;
	}

}

void SOCPACuda::calculate_sum_hypothesis_trace() {
	calculate_sum_hypothesis();

	// offload to GPU for sum_hypothesis_trace
	dim3 dimBlock = dim3(32, 32);
	dim3 dimGrid = dim3(byte_length, (NUM_GUESSES + dimBlock.x - 1) / dimBlock.x, (num_points + dimBlock.y - 1) / dimBlock.y);

	sum_hypothesis_trace_kernel<<<dimGrid, dimBlock>>>(byte_length, NUM_GUESSES, num_traces, num_points,
		device_hypothetial_leakage, device_traces, device_sum_hypothesis_trace);
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}

	// copy back
	CUDA_CHECK(cudaMemcpy((double*)sum_hypothesis_trace->get_pointer(),
							device_sum_hypothesis_trace,
							sum_hypothesis_trace->get_size(),
							cudaMemcpyDeviceToHost));
}


__global__
void sum_trace_kernel(int num_traces, int num_points, int window_size, double *traces,
	double *sum_trace_x_win, double *sum_trace2_x_win,
	double *sum_trace_x_win2, double *sum_trace2_x_win2)
{
	int point_index = blockIdx.x * blockDim.x + threadIdx.x;
	int window_index = blockIdx.y * blockDim.y + threadIdx.y;

	if (point_index + window_index + 1 < num_points && window_index < window_size) {
		double partial_sum_trace_x_win = 0;
		double partial_sum_trace2_x_win = 0;
		double partial_sum_trace_x_win2 = 0;
		double partial_sum_trace2_x_win2 = 0;
		for (int trace_index = 0; trace_index < num_traces; trace_index++) {
			double v1 = traces[trace_index * num_points + point_index];
			double v2 = traces[trace_index * num_points + point_index + window_index + 1];
			partial_sum_trace_x_win += v1 * v2;
			partial_sum_trace2_x_win += v1 * v1 * v2;
			partial_sum_trace_x_win2 += v1 * v2 * v2;
			partial_sum_trace2_x_win2 += v1 * v1 * v2 * v2;
		}
		sum_trace_x_win[point_index * window_size + window_index] = partial_sum_trace_x_win;
		sum_trace2_x_win[point_index * window_size + window_index] = partial_sum_trace2_x_win;
		sum_trace_x_win2[point_index * window_size + window_index] = partial_sum_trace_x_win2;
		sum_trace2_x_win2[point_index * window_size + window_index] = partial_sum_trace2_x_win2;
	}
}


void SOCPACuda::calculate_sum_trace()
{
	// SOCPA::update_sum_trace();
	dim3 dimBlock(32, 32);
	dim3 dimGrid((num_points + dimBlock.x - 1) / dimBlock.x,
				(window_size + dimBlock.y - 1) / dimBlock.y);
	sum_trace_kernel<<<dimGrid, dimBlock>>>(num_traces, num_points, window_size, device_traces,
		device_sum_trace_x_win, device_sum_trace2_x_win,
		device_sum_trace_x_win2, device_sum_trace2_x_win2);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int p = 0; p < num_points; p++) {
		for (int t = 0; t < num_traces; t++) {
			auto v1 = traces->at(t, p);
			sum_trace[p] += v1;
			sum_trace_square[p] += SQUARE(v1);
		}
	}
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}


	CUDA_CHECK(cudaMemcpy((double*)sum_trace_x_win->get_pointer(), device_sum_trace_x_win,
							sum_trace_x_win->get_size(),
							cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaMemcpy((double*)sum_trace2_x_win->get_pointer(), device_sum_trace2_x_win,
							sum_trace2_x_win->get_size(),
							cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((double*)sum_trace_x_win2->get_pointer(), device_sum_trace_x_win2,
							sum_trace_x_win2->get_size(),
							cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((double*)sum_trace2_x_win2->get_pointer(), device_sum_trace2_x_win2,
							sum_trace2_x_win2->get_size(),
							cudaMemcpyDeviceToHost));

}

const int trace_per_block = 16;
const int point_per_block = 32;
const int window_per_block = 16;

__global__
void sum_hypothesis_coumbined_trace_kernel(
	int num_traces, int start_point, int num_points, int window_size,
	int *hypothetial_leakage, double *traces,
	double *sum_hypothesis_combined_trace)
{
	__shared__ double trace_cache[trace_per_block][point_per_block + 1];
	__shared__ double hyp_cache[trace_per_block];

	int point_offset = blockIdx.z * blockDim.z + threadIdx.z;
	int point_index = point_offset + start_point;
	int trace_offset = blockIdx.x * trace_per_block;
	int end_trace = min(trace_per_block, num_traces - trace_offset);
	int end_window = min(window_size, num_points - point_index - 1);


	if (point_index < num_points) {
		// copy trace data to shared memory
		// assuming trace_per_block == window_per_block
		int trace_index = trace_offset + threadIdx.y;
		if (trace_index < num_traces) {
			trace_cache[threadIdx.y][threadIdx.z] = traces[trace_index * num_points + point_index];
		}
		if (threadIdx.z == 0) {
			hyp_cache[threadIdx.y] =  (trace_index < num_traces) ? (double)hypothetial_leakage[trace_index] : 0;
		}
		__syncthreads();

		for (int w = threadIdx.y; w < end_window; w += window_per_block) {
			double sum = 0;
			#pragma unroll
			for (int t = 0; t < end_trace; t++) {
				sum += hyp_cache[t] * trace_cache[t][threadIdx.z] * traces[(trace_offset + t) * num_points + point_index + w + 1];
			}
			atomicAdd(&sum_hypothesis_combined_trace[point_offset * window_size + w], sum);
		}
	}
}

__global__
void sum_hypothesis_coumbined_trace_kernel_nosm(
	int num_traces, int start_point, int num_points, int window_size,
	int *hypothetial_leakage, double *traces,
	double *sum_hypothesis_combined_trace)
{
	int point_offset = blockIdx.x * blockDim.x + threadIdx.x;
	int point_index = point_offset + start_point;
	int window_index = blockIdx.y * blockDim.y + threadIdx.y;

	double sum = 0;
	if (point_index < num_points && window_index < window_size && point_index + window_index + 1 < num_points) {
		for (int trace_index = 0; trace_index < num_traces; trace_index++) {
			int hyp = hypothetial_leakage[trace_index];
			sum += hyp * traces[trace_index * num_points + point_index] * traces[trace_index * num_points + point_index + window_index + 1];
		}
		sum_hypothesis_combined_trace[point_offset * window_size + window_index] = sum;
	}
}


void SOCPACuda::run_sum_hypothesis_coumbined_trace_kernel(int start_point, int hyp_offset)
{
	// x: traces, y: window, z: points
	dim3 dimBlock(1, window_per_block, point_per_block);
	dim3 dimGrid((num_traces + trace_per_block - 1) / trace_per_block,
				1,
				(point_tile_size + dimBlock.z - 1) / dimBlock.z);

	// clear sum_hypothesis_combined_trace on the GPU
	CUDA_CHECK(cudaMemset(device_sum_hypothesis_combined_trace, 0,
		sizeof(double) * point_tile_size * window_size));

	sum_hypothesis_coumbined_trace_kernel<<<dimGrid, dimBlock>>>(num_traces,
		start_point, num_points, window_size,
		&device_hypothetial_leakage[hyp_offset], device_traces,
		device_sum_hypothesis_combined_trace);

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}

	// copy back
	CUDA_CHECK(cudaMemcpy(sum_hypothesis_combined_trace,
							device_sum_hypothesis_combined_trace,
							sizeof(double) * point_tile_size * window_size,
							cudaMemcpyDeviceToHost));

}


void SOCPACuda::run_sum_hypothesis_coumbined_trace_kernel_nosm(int start_point, int hyp_offset)
{
	// x: traces, y: window, z: points
	dim3 dimBlock(32, 16);
	dim3 dimGrid((point_tile_size + dimBlock.x - 1) / dimBlock.x,
				(window_size + dimBlock.y - 1) / dimBlock.y);

	sum_hypothesis_coumbined_trace_kernel_nosm<<<dimGrid, dimBlock>>>(num_traces,
		start_point, num_points, window_size,
		&device_hypothetial_leakage[hyp_offset], device_traces,
		device_sum_hypothesis_combined_trace);

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}

	// copy back
	CUDA_CHECK(cudaMemcpy(sum_hypothesis_combined_trace,
							device_sum_hypothesis_combined_trace,
							sizeof(double) * point_tile_size * window_size,
							cudaMemcpyDeviceToHost));

}

void SOCPACuda::calculate_correlation_subkey(Array3D<RESULT_T>* corr)
{

	QUADFLOAT div_n = (QUADFLOAT)(1.0/(double)num_traces);
	QUADFLOAT div_nn = SQUARE(div_n);

	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int guess = 0; guess < NUM_GUESSES; guess++) {

			auto s3 = sum_hypothesis->at(byte_index, guess);
			auto s9 = sum_hypothesis_square->at(byte_index, guess);

			int hyp_offset = byte_index * NUM_GUESSES * num_traces + guess * num_traces;

			for (int tp = 0; tp < num_points; tp += point_tile_size) {

				if (use_shared_mem) {
					// run sum_hypothesis_coumbined_trace_kernel
					run_sum_hypothesis_coumbined_trace_kernel(tp, hyp_offset);
				} else {
					// run sum_hypothesis_coumbined_trace_kernel_nosm
					run_sum_hypothesis_coumbined_trace_kernel_nosm(tp, hyp_offset);
				}

				// calculate correlation
				#ifdef _OPENMP
				#pragma omp parallel for
				#endif
				for (int pp = 0; pp < point_tile_size; pp++) {
					int p = tp + pp;
					int end_window = std::min(window_size, num_points - p - 1);
					if (p < num_points) {
						RESULT_T max_corr = 0.0;
						int max_win = 0;
						auto s1 = (QUADFLOAT)sum_trace[p];
						auto s6 = (QUADFLOAT)sum_trace_square[p];
						auto s5 = sum_hypothesis_trace->at(byte_index, guess, p);
						for (int w = 0; w < end_window; w++) {
							auto s2 = (QUADFLOAT)sum_trace[p + w + 1];
							auto s8 = (QUADFLOAT)sum_trace_square[p + w + 1];
							auto s4 = (QUADFLOAT)sum_trace_x_win->at(p, w);
							auto s12 = (QUADFLOAT)sum_trace2_x_win->at(p, w);
							auto s13 = (QUADFLOAT)sum_trace_x_win2->at(p, w);
							auto s11 = (QUADFLOAT)sum_trace2_x_win2->at(p, w);
							auto s7 = sum_hypothesis_trace->at(byte_index, guess, p + w + 1);
							auto s10 = sum_hypothesis_combined_trace[pp * window_size + w];
							QUADFLOAT n_lambda3 = (QUADFLOAT)num_traces * s11 -
									QUADFLOAT(2.0) * (s2 * s12 + s1 * s13)  +
									(SQUARE(s2) * s6 + QUADFLOAT(4.0) * s1 * s2 * s4 + SQUARE(s1) * s8) * div_n -
									QUADFLOAT(3.0) * SQUARE(s1 * s2) * div_nn;
							QUADFLOAT lambda2 = s4 - (s1 * s2) * div_n;
							QUADFLOAT n_lambda1 = (QUADFLOAT)num_traces * s10 - (s1 * s7 + s2 * s5) + (s1 * s2 * s3) * div_n;
							RESULT_T corr = (RESULT_T)(n_lambda1 - lambda2 * s3) /
											std::sqrt((RESULT_T)(((n_lambda3 - SQUARE(lambda2)) * (num_traces * s9 - SQUARE(s3)))));

							if (std::abs(corr) > std::abs(max_corr)) {
								max_corr = corr;
								max_win = w;
							}
						} // end of window loop

						corr->at(byte_index, guess, p) = max_corr;
						max_combined_offset->at(byte_index, guess, p) = max_win;

					}
				} // end of partial point loop
			} // end of point tile loop
		} // end of guess loop
	} // end of byte loop
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

	if (device_traces == nullptr) {
		CUDA_CHECK(cudaMalloc((void**)&device_traces,
			traces->get_size()));
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
	cudaFree(device_sum_trace_x_win);
	cudaFree(device_sum_trace2_x_win);
	cudaFree(device_sum_trace_x_win2);
	cudaFree(device_sum_trace2_x_win2);
	delete[] sum_hypothesis_combined_trace;
};

PYBIND11_MODULE(socpa_cuda_kernel, module) {
	module.doc() = "CUDA implemetation plugin for SOCPA";

	py::class_<SOCPACuda,SOCPA>(module, "SOCPACuda")
		.def(py::init<int, int, int, AESLeakageModel::ModelBase*, bool>());

	py::class_<SOCPACudaFP32,SOCPA>(module, "SOCPACudaFP32")
		.def(py::init<int, int, int, AESLeakageModel::ModelBase*, bool>());

}

