/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPAOpenCLFP32.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  04-05-2025 06:37:15
*    Last Modified: 06-05-2025 08:11:14
*/

#include "SOCPAOpenCLFP32.hpp"

#include <string>

using namespace std;

#define OCL_FP32_PREMITIVES(...) #__VA_ARGS__

#define OCL_SUM_HYPOTHESIS_TRACE_FP32(...) #__VA_ARGS__
const char* SOCPAOpenCLFP32::sum_hypothesis_trace_kernel_code =
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS_TRACE_FP32

#define OCL_SUM_TRACE_FP32(...) #__VA_ARGS__
const char* SOCPAOpenCLFP32::sum_trace_kernel_code =
#include "device_code.cl"
;

#undef OCL_SUM_TRACE_FP32

#define OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32(...) #__VA_ARGS__
const char* SOCPAOpenCLFP32::sum_hypothesis_coumbined_trace_kernel_code =
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS_COMBINED_TRACE_FP32

#undef OCL_FP32_PREMITIVES

// =============================== Derived class ===============================

SOCPAOpenCLFP32::SOCPAOpenCLFP32(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model) : 
	SOCPAOpenCLBase(byte_length, num_points, window_size, model, false),
	traces_df64(nullptr), sum_hypothesis_trace_df64(nullptr),
	sum_trace_x_win_df64(nullptr), sum_trace2_x_win_df64(nullptr),
	sum_trace_x_win2_df64(nullptr), sum_trace2_x_win2_df64(nullptr)
{

	// create buffers
	allocate_device_memory();

	// create program
	create_programs();

	// build program
	build_kernel_programs();

	// create kernel
	create_kernels();

	clFinish(command_queue);

	// create buffers
	sum_hypothesis_combined_trace_df64 = new cl_float2[point_tile_size * window_size];

}

SOCPAOpenCLFP32::~SOCPAOpenCLFP32()
{
	// free device memory
	if (traces_df64 != nullptr) delete traces_df64;
	if (sum_hypothesis_trace_df64 != nullptr) delete sum_hypothesis_trace_df64;
	if (sum_trace_x_win_df64 != nullptr) delete sum_trace_x_win_df64;
	if (sum_trace2_x_win_df64 != nullptr) delete sum_trace2_x_win_df64;
	if (sum_trace_x_win2_df64 != nullptr) delete sum_trace_x_win2_df64;
	if (sum_trace2_x_win2_df64 != nullptr) delete sum_trace2_x_win2_df64;
	if (sum_hypothesis_combined_trace_df64 != nullptr) delete[]sum_hypothesis_combined_trace_df64;
}

void SOCPAOpenCLFP32::setup_arrays(py::array_t<TRACE_T> &py_traces,
	py::array_t<uint8_t> &py_plaintext,
	py::array_t<uint8_t> &py_ciphertext,
	py::array_t<uint8_t> &py_knownkey) {

	SOCPA::setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);

	cl_int err;
	// malloc gpu memory when not allocated
	if (cl_device_hypothetial_leakage == nullptr) {
		ALLOCATE_DEVICE_MEMORY(cl_device_hypothetial_leakage,
						CL_MEM_READ_ONLY,
						hypothetial_leakage->get_size());
	}
	if (cl_device_sum_hypothesis_trace == nullptr) {
		ALLOCATE_DEVICE_MEMORY(cl_device_sum_hypothesis_trace,
						CL_MEM_READ_WRITE,
						sum_hypothesis_trace->get_size());
		COPY_TO_DEVICE(cl_device_sum_hypothesis_trace,
			sum_hypothesis_trace->get_pointer(),
			sum_hypothesis_trace->get_size());
	}

	if (cl_device_traces == nullptr) {
		ALLOCATE_DEVICE_MEMORY(cl_device_traces,
					CL_MEM_READ_ONLY,
					traces->get_size());
	}
	// copy traces to GPU
	err = clFinish(command_queue);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to finish command queue ("
			+ to_string(err) + ")");
	}

	if (traces_df64 == nullptr) {
		traces_df64 = new Array2D<cl_float2>(num_traces, num_points);
	}
	if (sum_hypothesis_trace_df64 == nullptr) {
		sum_hypothesis_trace_df64 = new Array3D<cl_float2>(byte_length, NUM_GUESSES, num_points);
	}

	if (sum_trace_x_win_df64 == nullptr) {
		sum_trace_x_win_df64 = new Array2D<cl_float2>(num_points, window_size);
	}
	if (sum_trace2_x_win_df64 == nullptr) {
		sum_trace2_x_win_df64 = new Array2D<cl_float2>(num_points, window_size);
	}
	if (sum_trace_x_win2_df64 == nullptr) {
		sum_trace_x_win2_df64 = new Array2D<cl_float2>(num_points, window_size);
	}
	if (sum_trace2_x_win2_df64 == nullptr) {
		sum_trace2_x_win2_df64 = new Array2D<cl_float2>(num_points, window_size);;;
	}

	// convert double to float2
	double_to_float2(traces->get_pointer(), (cl_float2*)traces_df64->get_pointer(), num_traces * num_points);

	COPY_TO_DEVICE(cl_device_traces,
						traces_df64->get_pointer(),
						traces_df64->get_size());

	clFinish(command_queue);

}

void SOCPAOpenCLFP32::run_sum_trace()
{
	cl_int err;
	size_t work_size = min(sqrt_max_group_size, (size_t)16);
	size_t local_work_size[2] = {sqrt_max_group_size, sqrt_max_group_size};
	size_t ceiled_num_points = ((num_points + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];
	size_t ceiled_window_size = ((window_size + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];

	size_t global_work_size[2] = {(size_t)ceiled_num_points, (size_t)ceiled_window_size};

	clSetKernelArg(sum_trace_kernel, 0, sizeof(int), &num_traces);
	clSetKernelArg(sum_trace_kernel, 1, sizeof(int), &num_points);
	clSetKernelArg(sum_trace_kernel, 2, sizeof(int), &window_size);
	clSetKernelArg(sum_trace_kernel, 3, sizeof(cl_mem),
					&cl_device_traces);
	clSetKernelArg(sum_trace_kernel, 4, sizeof(cl_mem),
					&cl_device_sum_trace_x_win);
	clSetKernelArg(sum_trace_kernel, 5, sizeof(cl_mem),
					&cl_device_sum_trace2_x_win);
	clSetKernelArg(sum_trace_kernel, 6, sizeof(cl_mem),
					&cl_device_sum_trace_x_win2);
	clSetKernelArg(sum_trace_kernel, 7, sizeof(cl_mem),
					&cl_device_sum_trace2_x_win2);
	err = clEnqueueNDRangeKernel(command_queue, sum_trace_kernel,
								2, nullptr, global_work_size, local_work_size,
								0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		//get log
		size_t log_size;

		throw runtime_error("Error: Failed to execute kernel \"sum_trace_kernel\" ("
							+ to_string(err) + ")");
	}

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

	clFinish(command_queue);

	// copy back
	COPY_FROM_DEVICE(cl_device_sum_trace_x_win,
					(cl_float2*)sum_trace_x_win_df64->get_pointer(),
					sum_trace_x_win_df64->get_size());
	COPY_FROM_DEVICE(cl_device_sum_trace2_x_win,
					(cl_float2*)sum_trace2_x_win_df64->get_pointer(),
					sum_trace2_x_win_df64->get_size());
	COPY_FROM_DEVICE(cl_device_sum_trace_x_win2,
					(cl_float2*)sum_trace_x_win2_df64->get_pointer(),
					sum_trace_x_win2_df64->get_size());
	COPY_FROM_DEVICE(cl_device_sum_trace2_x_win2,
					(cl_float2*)sum_trace2_x_win2_df64->get_pointer(),
					sum_trace2_x_win2_df64->get_size());


	float2_to_double((cl_float2*)sum_trace_x_win_df64->get_pointer(), (TRACE_T*)sum_trace_x_win->get_pointer(), num_points * window_size);

	float2_to_double((cl_float2*)sum_trace2_x_win_df64->get_pointer(), (TRACE_T*)sum_trace2_x_win->get_pointer(), num_points * window_size);
	float2_to_double((cl_float2*)sum_trace_x_win2_df64->get_pointer(), (TRACE_T*)sum_trace_x_win2->get_pointer(), num_points * window_size);
	float2_to_double((cl_float2*)sum_trace2_x_win2_df64->get_pointer(), (TRACE_T*)sum_trace2_x_win2->get_pointer(), num_points * window_size);

	clFinish(command_queue);
}

void SOCPAOpenCLFP32::run_sum_hypothesis_trace_kernel()
{
	cl_int err;

	size_t local_guess_size = 8;
	size_t local_point_size = 32;
	// check if group size does not exceed the max group size
	while (local_guess_size * local_point_size > max_group_size) {
		local_point_size /= 2;
	}

	size_t local_work_size[3] = {1, local_guess_size, local_point_size};

	size_t ceiled_num_points = ((num_points + local_point_size - 1) / local_point_size) * local_point_size;

	size_t global_work_size[3] =
		{(size_t)byte_length, (size_t)NUM_GUESSES, (size_t)ceiled_num_points};

	clSetKernelArg(sum_hypothesis_trace_kernel, 0, sizeof(int), &byte_length);
	clSetKernelArg(sum_hypothesis_trace_kernel, 1, sizeof(int), &NUM_GUESSES);
	clSetKernelArg(sum_hypothesis_trace_kernel, 2, sizeof(int), &num_traces);
	clSetKernelArg(sum_hypothesis_trace_kernel, 3, sizeof(int), &num_points);
	clSetKernelArg(sum_hypothesis_trace_kernel, 4, sizeof(cl_mem),
					&cl_device_hypothetial_leakage);
	clSetKernelArg(sum_hypothesis_trace_kernel, 5, sizeof(cl_mem),
					&cl_device_traces);
	clSetKernelArg(sum_hypothesis_trace_kernel, 6, sizeof(cl_mem),
					&cl_device_sum_hypothesis_trace);

	err = clEnqueueNDRangeKernel(command_queue, sum_hypothesis_trace_kernel,
								3, nullptr, global_work_size, local_work_size,
								0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to execute kernel \"sum_hypothesis_trace_kernel\" ("
							+ to_string(err) + ")");
	}
	clFinish(command_queue);
	// copy back
	COPY_FROM_DEVICE(cl_device_sum_hypothesis_trace,
					(cl_float2*)sum_hypothesis_trace_df64->get_pointer(),
					sum_hypothesis_trace_df64->get_size());

	float2_to_double((cl_float2*)sum_hypothesis_trace_df64->get_pointer(), (RESULT_T*)sum_hypothesis_trace->get_pointer(), byte_length * NUM_GUESSES * num_points);


	clFinish(command_queue);

}

void SOCPAOpenCLFP32::run_sum_hypothesis_combined_trace_kernel(size_t *global_work_size, size_t *local_work_size)
{
	cl_int err;

	// clear sum_hypothesis_combined_trace on the GPU
	cl_float2 zero = {0.0f, 0.0f};
	err = clEnqueueFillBuffer(command_queue, cl_device_sum_hypothesis_combined_trace,
						&zero, sizeof(cl_float2), 0,
						sizeof(double) * point_tile_size * window_size, 0, nullptr, nullptr);

	err = clEnqueueNDRangeKernel(command_queue, sum_hypothesis_combined_trace_kernel,
								3, nullptr, global_work_size, local_work_size,
								0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to execute kernel \"sum_hypothesis_combined_trace_kernel\" ("
					+ to_string(err) + ")");
	}

	clFinish(command_queue);
	// copy back
	COPY_FROM_DEVICE(cl_device_sum_hypothesis_combined_trace,
					(cl_float2*)sum_hypothesis_combined_trace_df64,
					sizeof(cl_float2) * point_tile_size * window_size);
	// convert float2 to double
	float2_to_double((cl_float2*)sum_hypothesis_combined_trace_df64, (RESULT_T*)sum_hypothesis_combined_trace, point_tile_size * window_size);
}


void SOCPAOpenCLFP32::double_to_float2(const double *src, cl_float2 *dst, int length) {

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_float2(src[i]);
	}
}

void SOCPAOpenCLFP32::float2_to_double(const cl_float2 *src, double *dst, int length) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < length; i++) {
		dst[i] = to_double(src[i]);
	}
}