/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPACudaFP32.cu
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  09-05-2025 18:43:09
*    Last Modified: 23-05-2025 17:25:22
*/


#include "SOCPACudaFP32.hpp"
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

SOCPACudaFP32::SOCPACudaFP32(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model, bool use_shared_mem) :
	SOCPACuda(byte_length, num_points, window_size, model, use_shared_mem),
	traces_df64(nullptr), sum_hypothesis_trace_df64(nullptr),
	sum_trace_x_win_df64(nullptr), sum_trace2_x_win_df64(nullptr),
	sum_trace_x_win2_df64(nullptr), sum_trace2_x_win2_df64(nullptr)
{
	// allocate memory for the traces
	sum_hypothesis_combined_trace_df64 = new float2[point_tile_size * window_size];

}

SOCPACudaFP32::~SOCPACudaFP32()
{
	if (traces_df64 != nullptr)
		delete traces_df64;
	if (sum_hypothesis_trace_df64 != nullptr)
		delete sum_hypothesis_trace_df64;
	if (sum_trace_x_win_df64 != nullptr)
		delete sum_trace_x_win_df64;
	if (sum_trace2_x_win_df64 != nullptr)
		delete sum_trace2_x_win_df64;
	if (sum_trace_x_win2_df64 != nullptr)
		delete sum_trace_x_win2_df64;
	if (sum_trace2_x_win2_df64 != nullptr)
		delete sum_trace2_x_win2_df64;
	delete[] sum_hypothesis_combined_trace_df64;
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
		float2 sum = {0.0f, 0.0f};
		for (int trace_index = 0; trace_index < num_traces; trace_index++) {
			float hyp = (float)hypothetial_leakage[byte_index * num_guess * num_traces + guess * num_traces + trace_index];
			float2 trace = traces[trace_index * num_points + point_index];
			sum = df64_add(sum, df64_mul({hyp, 0.0f}, trace));
		}
		sum_hypothesis_trace[byte_index * num_guess * num_points + guess * num_points + point_index] = sum;
	}
}

void SOCPACudaFP32::calculate_sum_hypothesis_trace() {

	calculate_sum_hypothesis();

	// offload to GPU for sum_hypothesis_trace
	dim3 dimBlock = dim3(32, 32);
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


__global__
void sum_trace_kernel(int num_traces, int num_points, int window_size, float2 *traces,
	float2 *sum_trace_x_win, float2 *sum_trace2_x_win,
	float2 *sum_trace_x_win2, float2 *sum_trace2_x_win2)
{
	int point_index = blockIdx.x * blockDim.x + threadIdx.x;
	int window_index = blockIdx.y * blockDim.y + threadIdx.y;

	if (point_index + window_index + 1 < num_points && window_index < window_size) {
		float2 partial_sum_trace_x_win = {0, 0};
		float2 partial_sum_trace2_x_win = {0, 0};
		float2 partial_sum_trace_x_win2 = {0, 0};
		float2 partial_sum_trace2_x_win2 = {0, 0};
		for (int trace_index = 0; trace_index < num_traces; trace_index++) {
			float2 v1 = traces[trace_index * num_points + point_index];
			float2 v2 = traces[trace_index * num_points + point_index + window_index + 1];
			float2 v1v1 = df64_mul(v1, v1);
			float2 v2v2 = df64_mul(v2, v2);
			partial_sum_trace_x_win = df64_add(partial_sum_trace_x_win, df64_mul(v1, v2));
			partial_sum_trace2_x_win = df64_add(partial_sum_trace2_x_win, df64_mul(v1v1, v2));
			partial_sum_trace_x_win2 = df64_add(partial_sum_trace_x_win2, df64_mul(v2v2, v1));
			partial_sum_trace2_x_win2 = df64_add(partial_sum_trace2_x_win2, df64_mul(v1v1, v2v2));
		}
		sum_trace_x_win[point_index * window_size + window_index] = partial_sum_trace_x_win;
		sum_trace2_x_win[point_index * window_size + window_index] = partial_sum_trace2_x_win;
		sum_trace_x_win2[point_index * window_size + window_index] = partial_sum_trace_x_win2;
		sum_trace2_x_win2[point_index * window_size + window_index] = partial_sum_trace2_x_win2;
	}
}


void SOCPACudaFP32::calculate_sum_trace()
{
	// SOCPA::update_sum_trace();
	dim3 dimBlock(32, 16);
	dim3 dimGrid((num_points + dimBlock.x - 1) / dimBlock.x,
				(window_size + dimBlock.y - 1) / dimBlock.y);
	sum_trace_kernel<<<dimGrid, dimBlock>>>(num_traces, num_points, window_size, (float2*)device_traces,
		(float2*)device_sum_trace_x_win, (float2*)device_sum_trace2_x_win,
		(float2*)device_sum_trace_x_win2, (float2*)device_sum_trace2_x_win2);

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

	// copy back
	CUDA_CHECK(cudaMemcpy((double*)sum_trace_x_win_df64->get_pointer(), device_sum_trace_x_win,
							sum_trace_x_win_df64->get_size(),
							cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((double*)sum_trace2_x_win_df64->get_pointer(), device_sum_trace2_x_win,
							sum_trace2_x_win_df64->get_size(),
							cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((double*)sum_trace_x_win2_df64->get_pointer(), device_sum_trace_x_win2,
							sum_trace_x_win2_df64->get_size(),
							cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy((double*)sum_trace2_x_win2_df64->get_pointer(), device_sum_trace2_x_win2,
							sum_trace2_x_win2_df64->get_size(),
							cudaMemcpyDeviceToHost));

	// convert float2 to double
	float2_to_double((float2*)sum_trace_x_win_df64->get_pointer(), (TRACE_T*)sum_trace_x_win->get_pointer(), num_points * window_size);
	float2_to_double((float2*)sum_trace2_x_win_df64->get_pointer(), (TRACE_T*)sum_trace2_x_win->get_pointer(), num_points * window_size);
	float2_to_double((float2*)sum_trace_x_win2_df64->get_pointer(), (TRACE_T*)sum_trace_x_win2->get_pointer(), num_points * window_size);
	float2_to_double((float2*)sum_trace2_x_win2_df64->get_pointer(), (TRACE_T*)sum_trace2_x_win2->get_pointer(), num_points * window_size);

}

const int trace_per_block = 16;
const int point_per_block = 32;
const int window_per_block = 16;


__global__
void sum_hypothesis_coumbined_trace_kernel(
	int num_traces, int start_point, int num_points, int window_size,
	int *hypothetial_leakage, float2 *traces,
	float2 *sum_hypothesis_combined_trace)
{
	__shared__ float2 trace_cache[trace_per_block][point_per_block + 1];
	__shared__ float2 hyp_cache[trace_per_block];

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
			float hyp_float = (trace_index < num_traces) ? (float)hypothetial_leakage[trace_index] : 0;
			hyp_cache[threadIdx.y] = (float2){hyp_float, 0.0f};
		}
		__syncthreads();

		for (int w = threadIdx.y; w < end_window; w += window_per_block) {
			float2 sum = {0.0f, 0.0f};
			for (int t = 0; t < end_trace; t++) {
				float2 trace1 = trace_cache[t][threadIdx.z];
				float2 trace2 = traces[(trace_offset + t) * num_points + point_index + w + 1];
				float2 hyp = hyp_cache[t];
				sum = df64_add(sum, df64_mul(hyp, df64_mul(trace1, trace2)));
			}
			df64_atomic_add(&sum_hypothesis_combined_trace[point_offset * window_size + w], sum);
		}
	}
}

__global__
void sum_hypothesis_coumbined_trace_kernel_nosm(
	int num_traces, int start_point, int num_points, int window_size,
	int *hypothetial_leakage, float2 *traces,
	float2 *sum_hypothesis_combined_trace)
{
	int point_offset = blockIdx.x * blockDim.x + threadIdx.x;
	int point_index = point_offset + start_point;
	int window_index = blockIdx.y * blockDim.y + threadIdx.y;

	float2 sum = {0.0f, 0.0f};
	if (point_index < num_points && window_index < window_size && point_index + window_index + 1 < num_points) {
		for (int trace_index = 0; trace_index < num_traces; trace_index++) {
			float2 hyp = {(float)hypothetial_leakage[trace_index], 0.0f};
			float2 trace1 = traces[trace_index * num_points + point_index];
			float2 trace2 = traces[trace_index * num_points + point_index + window_index + 1];
			sum = df64_add(sum, df64_mul(hyp, df64_mul(trace1, trace2)));
		}
		sum_hypothesis_combined_trace[point_offset * window_size + window_index] = sum;
	}
}


void SOCPACudaFP32::run_sum_hypothesis_coumbined_trace_kernel(int start_point, int hyp_offset)
{
	// x: traces, y: window, z: points
	dim3 dimBlock(1, window_per_block, point_per_block);
	dim3 dimGrid((num_traces + trace_per_block - 1) / trace_per_block,
				1,
				(point_tile_size + dimBlock.z - 1) / dimBlock.z);

	// clear sum_hypothesis_combined_trace on the GPU
	CUDA_CHECK(cudaMemset(device_sum_hypothesis_combined_trace, 0,
		sizeof(float2) * point_tile_size * window_size));

	sum_hypothesis_coumbined_trace_kernel<<<dimGrid, dimBlock>>>(num_traces,
		start_point, num_points, window_size,
		&device_hypothetial_leakage[hyp_offset],
		(float2*)device_traces,
		(float2*)device_sum_hypothesis_combined_trace);

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}

	// copy back
	CUDA_CHECK(cudaMemcpy(sum_hypothesis_combined_trace_df64,
							device_sum_hypothesis_combined_trace,
							sizeof(float2) * point_tile_size * window_size,
							cudaMemcpyDeviceToHost));
	// convert float2 to double
	float2_to_double((float2*)sum_hypothesis_combined_trace_df64,
		(RESULT_T*)sum_hypothesis_combined_trace, point_tile_size * window_size);

}


void SOCPACudaFP32::run_sum_hypothesis_coumbined_trace_kernel_nosm(int start_point, int hyp_offset)
{
	// x: traces, y: window, z: points
	dim3 dimBlock(32, 16);
	dim3 dimGrid((point_tile_size + dimBlock.x - 1) / dimBlock.x,
				(window_size + dimBlock.y - 1) / dimBlock.y);

	sum_hypothesis_coumbined_trace_kernel_nosm<<<dimGrid, dimBlock>>>(num_traces,
		start_point, num_points, window_size,
		&device_hypothetial_leakage[hyp_offset],
		(float2*)device_traces,
		(float2*)device_sum_hypothesis_combined_trace);

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA Error ") + cudaGetErrorString(err));
	}

	// copy back
	CUDA_CHECK(cudaMemcpy(sum_hypothesis_combined_trace_df64,
							device_sum_hypothesis_combined_trace,
							sizeof(float2) * point_tile_size * window_size,
							cudaMemcpyDeviceToHost));
	// convert float2 to double
	float2_to_double(sum_hypothesis_combined_trace_df64,
		(RESULT_T*)sum_hypothesis_combined_trace, point_tile_size * window_size);
}



void SOCPACudaFP32::setup_arrays(py::array_t<TRACE_T> &py_traces,
	py::array_t<uint8_t> &py_plaintext,
	py::array_t<uint8_t> &py_ciphertext,
	py::array_t<uint8_t> &py_knownkey)
{

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

	// allocate memory for format conversion
	if (traces_df64 == nullptr) {
		traces_df64 = new Array2D<float2>(num_traces, num_points);
	}
	if (sum_hypothesis_trace_df64 == nullptr) {
		sum_hypothesis_trace_df64 = new Array3D<float2>(byte_length, NUM_GUESSES, num_points);
	}

	if (sum_trace_x_win_df64 == nullptr) {
		sum_trace_x_win_df64 = new Array2D<float2>(num_points, window_size);
	}
	if (sum_trace2_x_win_df64 == nullptr) {
		sum_trace2_x_win_df64 = new Array2D<float2>(num_points, window_size);
	}
	if (sum_trace_x_win2_df64 == nullptr) {
		sum_trace_x_win2_df64 = new Array2D<float2>(num_points, window_size);
	}
	if (sum_trace2_x_win2_df64 == nullptr) {
		sum_trace2_x_win2_df64 = new Array2D<float2>(num_points, window_size);;;
	}


	// convert double to float2
	double_to_float2(traces->get_pointer(), (float2*)traces_df64->get_pointer(), num_traces * num_points);

	// copy trace data
	CUDA_CHECK(cudaMemcpy(device_traces, traces_df64->get_pointer(),
			traces_df64->get_size(),
			cudaMemcpyHostToDevice));

}

void SOCPACudaFP32::double_to_float2(const double *src, float2 *dst, int length) {

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_float2(src[i]);
	}
}

void SOCPACudaFP32::float2_to_double(const float2 *src, double *dst, int length) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_double(src[i]);
	}
}