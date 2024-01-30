/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPAOpenCL.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2024 12:31:30
*    Last Modified: 30-01-2024 12:31:30
*/


#include "FastCPAOpenCL.hpp"
#include "FastCPAOpenCLFP32.hpp"

#include <string>

using namespace std;

#define OCL_SUM_HYPOTHESIS(x) #x
const char* FastCPAOpenCLBase::sum_hypothesis_kernel_code =
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS

#define OCL_SUM_HYPOTHESIS_TRACE(...) #__VA_ARGS__
const char* FastCPAOpenCLBase::sum_hypothesis_trace_kernel_code = 
#include "device_code.cl"
;
#undef OCL_SUM_HYPOTHESIS_TRACE

// =============================== Base class ===============================

FastCPAOpenCLBase::FastCPAOpenCLBase(int num_traces, int num_points, AESLeakageModel::ModelBase *model) : FastCPA(num_traces, num_points, model),
	cl_device_traces(nullptr), cl_device_hypothetial_leakage(nullptr),
	cl_device_sum_hypothesis_trace(nullptr)
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

	// copy initial array data to allocated device memory
	COPY_TO_DEVICE(cl_device_sum_hypothesis, sum_hypothesis->get_pointer(),
					sum_hypothesis->get_size());
	COPY_TO_DEVICE(cl_device_sum_hypothesis_square,
					sum_hypothesis_square->get_pointer(),
					sum_hypothesis_square->get_size());

}

FastCPAOpenCLBase::~FastCPAOpenCLBase()
{
	// release memory
	clReleaseMemObject(cl_device_traces);
	clReleaseMemObject(cl_device_hypothetial_leakage);
	clReleaseMemObject(cl_device_sum_hypothesis);
	clReleaseMemObject(cl_device_sum_hypothesis_square);
	clReleaseMemObject(cl_device_sum_hypothesis_trace);
	// release kernel
	clReleaseKernel(sum_hypothesis_kernel);
	clReleaseKernel(sum_hypothesis_trace_kernel);
	clReleaseProgram(sum_hypothesis_kernel_program);
	clReleaseProgram(sum_hypothesis_trace_kernel_program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

cl_platform_id FastCPAOpenCLBase::get_target_platform()
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
		fprintf(stderr, "Target platform is specified by environment variable CL_PLATFORM %s\n", env);
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

cl_device_id FastCPAOpenCLBase::get_target_device(cl_platform_id platform)
{
	cl_int err;
	cl_device_id target_device;
	cl_uint deviceCount;
	// get device count
	vector<cl_device_id> devices(deviceCount);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device count ("
							+ to_string(err) + ")");
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to get device IDs ("
							+ to_string(err) + ")");
	}
	// get environment variables
	const char *env = getenv("CL_DEVICE");
	if (env != nullptr) {
		fprintf(stderr, "Target device is specified by environment variable CL_DEVICE %s\n", env);
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

void FastCPAOpenCLBase::calculate_hypothesis()
{
	FastCPA::calculate_hypothesis();
	// copy CPU calculated hypothetial_leakage to GPU
	COPY_TO_DEVICE(cl_device_hypothetial_leakage,
					hypothetial_leakage->get_pointer(),
					hypothetial_leakage->get_size());
	clFinish(command_queue);
}


void FastCPAOpenCLBase::run_sum_hypothesis_kernel()
{
	cl_int err;
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
								2, nullptr, global_work_size, nullptr,
								0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to execute kernel \"sum_hypothesis_kernel\" ("
							+ to_string(err) + ")");
	}
	clFinish(command_queue);
}

void FastCPAOpenCLBase::run_sum_hypothesis_trace_kernel()
{
	cl_int err;
	size_t global_work_size[3] =
		{(size_t)byte_length, (size_t)NUM_GUESSES, (size_t)num_points};

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
								3, nullptr, global_work_size, nullptr,
								0, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		throw runtime_error("Error: Failed to execute kernel \"sum_hypothesis_trace_kernel\" ("
							+ to_string(err) + ")");
	}
	clFinish(command_queue);
}


// =============================== Derived class ===============================

FastCPAOpenCL::FastCPAOpenCL(int num_traces, int num_points, AESLeakageModel::ModelBase *model) : 
	FastCPAOpenCLBase(num_traces, num_points, model)
{
	cl_int err;

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

	clFinish(command_queue);

}


void FastCPAOpenCL::setup_arrays(py::array_t<double> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey) {
	FastCPA::setup_arrays(py_traces, py_plaintext, py_ciphertext, py_knownkey);

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


void FastCPAOpenCL::calculate_correlation_subkey(Array3D<double>* diff, QUADFLOAT *sumden2) {


	// offload to GPU for sum_hypothesis, sum_hypothesis_square
	run_sum_hypothesis_kernel();

	// offload to GPU for sum_hypothesis_trace
	run_sum_hypothesis_trace_kernel();

	// copy back
	COPY_FROM_DEVICE(cl_device_sum_hypothesis,
					(int64_t*)sum_hypothesis->get_pointer(),
					sum_hypothesis->get_size());
	COPY_FROM_DEVICE(cl_device_sum_hypothesis_square,
					(int64_t*)sum_hypothesis_square->get_pointer(),
					sum_hypothesis_square->get_size());
	COPY_FROM_DEVICE(cl_device_sum_hypothesis_trace,
					(double*)sum_hypothesis_trace->get_pointer(),
					sum_hypothesis_trace->get_size());
	clFinish(command_queue);

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
				diff->at(byte_index, guess, p) = (double)sumnum / std::sqrt((double)sumden1 * (double)sumden2[p]);
			}
		}
	}
}

PYBIND11_MODULE(cpa_opencl_kernel, module) {
	module.doc() = "OpenCL implemetation plugin for CPA";
	py::class_<FastCPAOpenCL,FastCPA>(module, "FastCPAOpenCL")
		.def(py::init<int, int, AESLeakageModel::ModelBase*>());

	py::class_<FastCPAOpenCLFP32,FastCPA>(module, "FastCPAOpenCLFP32")
		.def(py::init<int, int, AESLeakageModel::ModelBase*>());

}