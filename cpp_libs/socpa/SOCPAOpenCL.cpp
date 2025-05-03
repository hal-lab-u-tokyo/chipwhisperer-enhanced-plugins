/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPAOpenCL.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  03-05-2025 05:56:52
*    Last Modified: 03-05-2025 10:17:12
*/


#include "SOCPAOpenCL.hpp"

#include <string>

using namespace std;

#define OCL_SUM_HYPOTHESIS(x) #x
const char* SOCPAOpenCLBase::sum_hypothesis_kernel_code =
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS

#define OCL_SUM_HYPOTHESIS_TRACE(...) #__VA_ARGS__
const char* SOCPAOpenCLBase::sum_hypothesis_trace_kernel_code =
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS_TRACE

#define OCL_SUM_TRACE(...) #__VA_ARGS__
const char* SOCPAOpenCLBase::sum_trace_kernel_code =
#include "device_code.cl"
;
#undef OCL_SUM_TRACE

#define OCL_SUM_HYPOTHESIS_COMBINED_TRACE(...) #__VA_ARGS__
const char* SOCPAOpenCLBase::sum_hypothesis_coumbined_trace_kernel_code =
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS_COMBINED_TRACE


// =============================== Base class ===============================

SOCPAOpenCLBase::SOCPAOpenCLBase(int byte_length, int num_points, int window_size,
	AESLeakageModel::ModelBase *model)
	: SOCPA(byte_length, num_points, window_size, model),
	cl_device_traces(nullptr), cl_device_hypothetial_leakage(nullptr),
	cl_device_sum_hypothesis_trace(nullptr),
	sum_trace_kernel(nullptr), sum_hypothesis_combined_trace_kernel(nullptr),
	sum_hypothesis_kernel(nullptr), sum_hypothesis_trace_kernel(nullptr),
	sum_trace_kernel_program(nullptr), sum_hypothesis_combined_trace_kernel_program(nullptr),
	sum_hypothesis_kernel_program(nullptr), sum_hypothesis_trace_kernel_program(nullptr)
{

	cl_int err;
	// get platform
	platform_id = get_target_platform();
	// get device
	device_id = get_target_device(platform_id);

	// create context
	context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create context ("
							+ to_string(err) + ")");
	}

	// get device memory size
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
							sizeof(global_mem_size), &global_mem_size, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device info ("
							+ to_string(err) + ")");
	}
	err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE,
							sizeof(local_mem_size), &local_mem_size, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device info ("
							+ to_string(err) + ")");
	}

	// determine tile size no to ocupy more than 50% of the global memory for the temporary array
	point_tile_size = 1 << static_cast<int>(std::ceil(std::log2(num_points)));

	while ((sizeof(double) * point_tile_size * num_points) > (global_mem_size / 2)) {
		point_tile_size /= 2;
	}

	point_tile_size = std::min(point_tile_size, num_points);

	printf("Device global memory size: %zu\n", global_mem_size);
	printf("Device local memory size: %zu\n", local_mem_size);

	// create command queue
#ifdef __APPLE__
	// because supported only in OpenCL 1.2
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
#else
	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
#endif

	// create buffers
	ALLOCATE_DEVICE_MEMORY(cl_device_sum_hypothesis, CL_MEM_READ_WRITE,
						sum_hypothesis->get_size());
	ALLOCATE_DEVICE_MEMORY(cl_device_sum_hypothesis_square,
							CL_MEM_READ_WRITE,
							sum_hypothesis_square->get_size());
	ALLOCATE_DEVICE_MEMORY(cl_device_sum_trace_x_win,
							CL_MEM_READ_WRITE,
							sum_trace_x_win->get_size());
	ALLOCATE_DEVICE_MEMORY(cl_device_sum_trace2_x_win,
							CL_MEM_READ_WRITE,
							sum_trace2_x_win->get_size());
	ALLOCATE_DEVICE_MEMORY(cl_device_sum_trace_x_win2,
							CL_MEM_READ_WRITE,
							sum_trace_x_win2->get_size());
	ALLOCATE_DEVICE_MEMORY(cl_device_sum_trace2_x_win2,
							CL_MEM_READ_WRITE,
							sum_trace2_x_win2->get_size());

	ALLOCATE_DEVICE_MEMORY(cl_device_sum_hypothesis_combined_trace,
							CL_MEM_READ_WRITE,
							sizeof(double) * point_tile_size * window_size);

}

SOCPAOpenCLBase::~SOCPAOpenCLBase()
{
	// release memory if allocated
	if (cl_device_traces != nullptr) clReleaseMemObject(cl_device_traces);
	if (cl_device_hypothetial_leakage != nullptr) clReleaseMemObject(cl_device_hypothetial_leakage);
	if (cl_device_sum_hypothesis_trace != nullptr) clReleaseMemObject(cl_device_sum_hypothesis_trace);
	if (cl_device_sum_hypothesis != nullptr) clReleaseMemObject(cl_device_sum_hypothesis);
	if (cl_device_sum_hypothesis_square != nullptr) clReleaseMemObject(cl_device_sum_hypothesis_square);
	if (cl_device_sum_trace_x_win != nullptr) clReleaseMemObject(cl_device_sum_trace_x_win);
	if (cl_device_sum_trace2_x_win != nullptr) clReleaseMemObject(cl_device_sum_trace2_x_win);
	if (cl_device_sum_trace_x_win2 != nullptr) clReleaseMemObject(cl_device_sum_trace_x_win2);
	if (cl_device_sum_trace2_x_win2 != nullptr) clReleaseMemObject(cl_device_sum_trace2_x_win2);
	if (cl_device_sum_hypothesis_combined_trace != nullptr) clReleaseMemObject(cl_device_sum_hypothesis_combined_trace);

	// release kernel iff kernel is created
	if (sum_hypothesis_kernel != nullptr) clReleaseKernel(sum_hypothesis_kernel);
	if (sum_hypothesis_trace_kernel != nullptr) clReleaseKernel(sum_hypothesis_trace_kernel);
	if (sum_trace_kernel != nullptr) clReleaseKernel(sum_trace_kernel);
	if (sum_hypothesis_combined_trace_kernel != nullptr) clReleaseKernel(sum_hypothesis_combined_trace_kernel);

	// release program iff program is created
	if (sum_hypothesis_kernel_program != nullptr) clReleaseProgram(sum_hypothesis_kernel_program);
	if (sum_hypothesis_trace_kernel_program != nullptr) clReleaseProgram(sum_hypothesis_trace_kernel_program);
	if (sum_trace_kernel_program != nullptr) clReleaseProgram(sum_trace_kernel_program);
	if (sum_hypothesis_combined_trace_kernel_program != nullptr) clReleaseProgram(sum_hypothesis_combined_trace_kernel_program);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

cl_platform_id SOCPAOpenCLBase::get_target_platform()
{
	cl_int err;
	cl_uint platformCount;
	// Get the number of platforms
	err = clGetPlatformIDs(0, nullptr, &platformCount);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get platform count ("
							+ to_string(err) + ")");
	}
	vector<cl_platform_id> platforms(platformCount);
	err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get platform IDs ("
							 + to_string(err) + ")");
	}

	cl_platform_id platform;
	// get environment variables
	const char *env = getenv("CL_PLATFORM");
	if (env != nullptr) {
		int select_id = stoi(env);
		if (select_id >= platformCount) {
			throw runtime_error("Error: Invalid platform ID");
		}
		platform = platforms[select_id];
	} else {
		platform = platforms[0];
	}

	return platform;
}

cl_device_id SOCPAOpenCLBase::get_target_device(cl_platform_id platform)
{
	cl_int err;
	cl_device_id target_device;
	cl_uint deviceCount;
	// get device count
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device count ("
							+ to_string(err) + ")");
	}
	vector<cl_device_id> devices(deviceCount);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device IDs ("
							+ to_string(err) + ")");
	}
	// get environment variables
	const char *env = getenv("CL_DEVICE");
	if (env != nullptr) {
		int select_id = stoi(env);
		if (select_id >= deviceCount) {
			throw runtime_error("Error: Invalid device ID");
		}
		target_device = devices[select_id];
	} else {
		target_device = devices[0];
	}

	return target_device;
}

void SOCPAOpenCLBase::calculate_hypothesis()
{
	SOCPA::calculate_hypothesis();
	// copy CPU calculated hypothetial_leakage to GPU
	COPY_TO_DEVICE(cl_device_hypothetial_leakage,
					hypothetial_leakage->get_pointer(),
					hypothetial_leakage->get_size());
	clFinish(command_queue);
}


void SOCPAOpenCLBase::run_sum_hypothesis_kernel()
{
	cl_int err;
	size_t local_work_size[2] = {4, 8};
	size_t global_work_size[2] = {(size_t)byte_length, (size_t)(NUM_GUESSES)};

	// set kernel arguments
	clSetKernelArg(sum_hypothesis_kernel, 0, sizeof(int), &byte_length);
	clSetKernelArg(sum_hypothesis_kernel, 1, sizeof(int), &NUM_GUESSES);
	clSetKernelArg(sum_hypothesis_kernel, 2, sizeof(int), &num_traces);
	clSetKernelArg(sum_hypothesis_kernel, 3, sizeof(cl_mem),
					&cl_device_hypothetial_leakage);
	clSetKernelArg(sum_hypothesis_kernel, 4, sizeof(cl_mem),
					&cl_device_sum_hypothesis);
	clSetKernelArg(sum_hypothesis_kernel, 5, sizeof(cl_mem),
					&cl_device_sum_hypothesis_square);
	err = clEnqueueNDRangeKernel(command_queue, sum_hypothesis_kernel,
								2, nullptr, global_work_size, local_work_size,
								0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to execute kernel \"sum_hypothesis_kernel\" ("
							+ to_string(err) + ")");
	}
	clFinish(command_queue);

	COPY_FROM_DEVICE(cl_device_sum_hypothesis,
		(int64_t*)sum_hypothesis->get_pointer(),
		sum_hypothesis->get_size());
	COPY_FROM_DEVICE(cl_device_sum_hypothesis_square,
		(int64_t*)sum_hypothesis_square->get_pointer(),
		sum_hypothesis_square->get_size());
	clFinish(command_queue);
}

void SOCPAOpenCLBase::calculate_sum_hypothesis_trace()
{
	run_sum_hypothesis_kernel();

	cl_int err;

	const int local_guess_size = 8;
	const int local_point_size = 32;
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
					(RESULT_T*)sum_hypothesis_trace->get_pointer(),
					sum_hypothesis_trace->get_size());

	clFinish(command_queue);
}

void SOCPAOpenCLBase::calculate_sum_trace()
{

	cl_int err;
	size_t local_work_size[2] = {32, 32};
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
					(TRACE_T*)sum_trace_x_win->get_pointer(),
					sum_trace_x_win->get_size());
	COPY_FROM_DEVICE(cl_device_sum_trace2_x_win,
					(TRACE_T*)sum_trace2_x_win->get_pointer(),
					sum_trace2_x_win->get_size());
	COPY_FROM_DEVICE(cl_device_sum_trace_x_win2,
					(TRACE_T*)sum_trace_x_win2->get_pointer(),
					sum_trace_x_win2->get_size());
	COPY_FROM_DEVICE(cl_device_sum_trace2_x_win2,
					(TRACE_T*)sum_trace2_x_win2->get_pointer(),
					sum_trace2_x_win2->get_size());
	clFinish(command_queue);
}

const int trace_per_block = 16;
const int point_per_block = 32;
const int window_per_block = 16;

#include <chrono>
void SOCPAOpenCLBase::calculate_correlation_subkey(Array3D<RESULT_T>* corr)
{

	cl_int err;
	// threads size
	size_t local_work_size[3] = {1, window_per_block, point_per_block};
	size_t ceiled_num_points = ((point_tile_size + point_per_block - 1) / point_per_block) * point_per_block;
	size_t global_work_size[3] = {(size_t)(num_traces + trace_per_block - 1) / trace_per_block,
								 window_per_block, (size_t)ceiled_num_points};

	// set common kernel arguments
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 0, sizeof(int), &num_traces);
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 2, sizeof(int), &num_points);
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 3, sizeof(int), &window_size);
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 5, sizeof(double) * trace_per_block * (point_per_block + 1), NULL);
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 6, sizeof(double) * trace_per_block, NULL);
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 7, sizeof(cl_mem),
	&cl_device_hypothetial_leakage);
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 8, sizeof(cl_mem),
	&cl_device_traces);
	clSetKernelArg(sum_hypothesis_combined_trace_kernel, 9, sizeof(cl_mem),

	// check if local memory size is enough
	&cl_device_sum_hypothesis_combined_trace);
	size_t required_local_mem_size = (sizeof(double) * trace_per_block * (point_per_block + 1)) + (sizeof(int) * trace_per_block);
	if (required_local_mem_size > local_mem_size) {
		throw runtime_error("Error: Local memory size is not enough");
	}

	// temporary storage
	auto sum_hypothesis_combined_trace = new double[point_tile_size * window_size];
	for (int byte_index = 0; byte_index < byte_length; byte_index++) {
		for (int guess = 0; guess < NUM_GUESSES; guess++) {

			auto s3 = sum_hypothesis->at(byte_index, guess);
			auto s9 = sum_hypothesis_square->at(byte_index, guess);
			int hyp_offset = byte_index * NUM_GUESSES * num_traces + guess * num_traces;
			clSetKernelArg(sum_hypothesis_combined_trace_kernel, 4, sizeof(int), &hyp_offset);

			for (int tp = 0; tp < num_points; tp += point_tile_size) {

				int start_point = tp;

				// clear sum_hypothesis_combined_trace on the GPU
				double zero = 0.0;
				err = clEnqueueFillBuffer(command_queue, cl_device_sum_hypothesis_combined_trace,
								&zero, sizeof(double), 0,
								sizeof(double) * point_tile_size * window_size, 0, nullptr, nullptr);
				clSetKernelArg(sum_hypothesis_combined_trace_kernel, 1, sizeof(int), &start_point);

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
					(RESULT_T*)sum_hypothesis_combined_trace,
					sizeof(double) * point_tile_size * window_size);

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
							QUADFLOAT n_lambda3 = (QUADFLOAT)num_traces * s11 
									- 2.0 * (s2 * s12 + s1 * s13)  +
									(SQUARE(s2) * s6 + 4.0 * s1 * s2 * s4 + SQUARE(s1) * s8) / (QUADFLOAT)num_traces -
									3.0 * SQUARE(s1 * s2) / (QUADFLOAT)SQUARE(num_traces);
							QUADFLOAT lambda2 = s4 - (s1 * s2)/(QUADFLOAT)num_traces;
							QUADFLOAT n_lambda1 = (QUADFLOAT)num_traces * s10 - (s1 * s7 + s2 * s5) + (s1 * s2 * s3)/ (QUADFLOAT)num_traces;
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

// =============================== Derived class ===============================

SOCPAOpenCL::SOCPAOpenCL(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model) :
	SOCPAOpenCLBase(byte_length, num_points, window_size, model)
{

	cl_int err;

	if(!is_double_available(device_id)) {
		throw runtime_error("Error: Double precision is not supported\n"
							"Try to use SOCPAOpenCLFP32 instead");
	}

	// create program
	sum_hypothesis_kernel_program =
		clCreateProgramWithSource(context, 1, get_sum_hypothesis_kernel_code(), nullptr, &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create program ("
							+ to_string(err) + ")");
	}
	sum_hypothesis_trace_kernel_program =
		clCreateProgramWithSource(context, 1, get_sum_hypothesis_trace_kernel_code(), nullptr, &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create program ("
							+ to_string(err) + ")");
	}

	sum_trace_kernel_program =
		clCreateProgramWithSource(context, 1, get_sum_trace_kernel_code(), nullptr, &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create program ("
							+ to_string(err) + ")");
	}

	sum_hypothesis_combined_trace_kernel_program =
		clCreateProgramWithSource(context, 1, get_sum_hypothesis_combined_trace_kernel_code(), nullptr, &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create program ("
							+ to_string(err) + ")");
	}

	// build program
	err = clBuildProgram(sum_hypothesis_kernel_program, 1, &device_id, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to build program \"sum_hypothesis_kernel\" ("
							+ to_string(err) + ")");
	}
	err = clBuildProgram(sum_hypothesis_trace_kernel_program, 1, &device_id, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
			throw runtime_error("Error: Failed to build program \"sum_hypothesis_trace_kernel\" ("
							+ to_string(err) + ")");
	}
	err = clBuildProgram(sum_trace_kernel_program, 1, &device_id, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to build program \"sum_trace_kernel\" ("
							+ to_string(err) + ")");
	}

	err = clBuildProgram(sum_hypothesis_combined_trace_kernel_program, 1, &device_id, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to build program \"sum_hypothesis_combined_trace_kernel\" ("
							+ to_string(err) + ")");
	}

	// create kernel
	sum_hypothesis_kernel = clCreateKernel(sum_hypothesis_kernel_program,
								"sum_hypothesis_kernel", &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create kernel \"sum_hypothesis_kernel\" ("
							+ to_string(err) + ")");
	}

	sum_hypothesis_trace_kernel
		= clCreateKernel(sum_hypothesis_trace_kernel_program,
								"sum_hypothesis_trace_kernel", &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create kernel \"sum_hypothesis_trace_kernel\" ("
							+ to_string(err) + ")");
	}

	sum_trace_kernel
		= clCreateKernel(sum_trace_kernel_program,
								"sum_trace_kernel", &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create kernel \"sum_trace_kernel\" ("
							+ to_string(err) + ")");
	}

	sum_hypothesis_combined_trace_kernel
		= clCreateKernel(sum_hypothesis_combined_trace_kernel_program,
								"sum_hypothesis_combined_trace_kernel", &err);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to create kernel \"sum_hypothesis_combined_trace_kernel\" ("
							+ to_string(err) + ")");
	}

	clFinish(command_queue);

}


void SOCPAOpenCL::setup_arrays(py::array_t<TRACE_T> &py_traces,
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

	COPY_TO_DEVICE(cl_device_traces,
					traces->get_pointer(),
					traces->get_size());
	clFinish(command_queue);
}




bool SOCPAOpenCL::is_double_available(cl_device_id device_id)
{
	cl_int err;
	size_t ext_size;
	err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device info ("
							+ to_string(err) + ")");
	}
	string ext(ext_size, '\0');
	err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, ext_size, &ext[0], nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device info ("
							+ to_string(err) + ")");
	}
	size_t pos = ext.find("cl_khr_fp64");
	return (pos != string::npos);
}

PYBIND11_MODULE(socpa_opencl_kernel, module) {
	module.doc() = "OpenCL implemetation plugin for SOCPA";
	py::class_<SOCPAOpenCL,SOCPA>(module, "SOCPAOpenCL")
		.def(py::init<int, int, int, AESLeakageModel::ModelBase*>());

	// py::class_<FastCPAOpenCLFP32,FastCPA>(module, "FastCPAOpenCLFP32")
	// 	.def(py::init<int, int, AESLeakageModel::ModelBase*>());

}