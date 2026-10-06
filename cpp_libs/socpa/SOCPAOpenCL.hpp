/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPAOpenCL.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  03-05-2025 05:56:44
*    Last Modified: 07-05-2025 15:17:47
*/

#ifndef SOCPAOPENCL_H
#define SOCPAOPENCL_H

#include "SOCPA.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

#define ALLOCATE_DEVICE_MEMORY(cl_mem, flag, size) \
{	cl_int alloc_err; \
	cl_mem = clCreateBuffer(context, flag, size, NULL, &alloc_err); \
	if (alloc_err != CL_SUCCESS) { \
		throw runtime_error("Error: Failed to create buffer at line " + \
							 to_string(__LINE__) + " (" \
							+ to_string(alloc_err) + ")"); \
	} \
}

#define COPY_TO_DEVICE(cl_mem, host_mem, size) \
{	cl_int copy_err; \
	copy_err = clEnqueueWriteBuffer(command_queue, cl_mem, CL_TRUE, 0, size, host_mem, 0, nullptr, nullptr); \
	if (copy_err != CL_SUCCESS) { \
		throw runtime_error("Error: Failed to copy to device at " + \
							to_string(__LINE__) + "(" \
							+ to_string(copy_err) + ")"); \
	} \
}

#define COPY_FROM_DEVICE(cl_mem, host_mem, size) \
{	cl_int copy_err; \
	copy_err = clEnqueueReadBuffer(command_queue, cl_mem, CL_TRUE, 0, size, host_mem, 0, nullptr, nullptr); \
	if (copy_err != CL_SUCCESS) { \
		throw runtime_error("Error: Failed to copy from device at " + \
							to_string(__LINE__) + "(" \
							+ to_string(copy_err) + ")"); \
	} \
}

class SOCPAOpenCLBase : public SOCPA
{
public:
	SOCPAOpenCLBase(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model, bool need_double, bool use_shared_mem);
	~SOCPAOpenCLBase();
protected:
	// OpenCL device memory
	// host -> device
	cl_mem cl_device_traces, cl_device_hypothetial_leakage;
	// device -> host
	cl_mem cl_device_sum_hypothesis,
			cl_device_sum_hypothesis_square, cl_device_sum_hypothesis_trace,
			cl_device_sum_trace_x_win, cl_device_sum_trace2_x_win, cl_device_sum_trace_x_win2, cl_device_sum_trace2_x_win2;
	cl_mem cl_device_sum_hypothesis_combined_trace;

	cl_context context;
	cl_command_queue command_queue;
	// OpenCL kernel programs
	cl_program sum_trace_kernel_program;
	cl_program sum_hypothesis_kernel_program;
	cl_program sum_hypothesis_trace_kernel_program;
	cl_program sum_hypothesis_combined_trace_kernel_program;

	// OpenCL kernel objects
	cl_kernel sum_hypothesis_kernel, sum_hypothesis_trace_kernel, sum_trace_kernel, sum_hypothesis_combined_trace_kernel;

	cl_platform_id platform_id;
	cl_device_id device_id;

	// device param
	size_t local_mem_size;
	size_t global_mem_size;
	size_t max_group_size;
	size_t sqrt_max_group_size;

	// intermediate array
	double *sum_hypothesis_combined_trace;

	virtual const char** get_sum_trace_kernel_code() { return &sum_trace_kernel_code; }
	virtual const char** get_sum_hypothesis_kernel_code() { return &sum_hypothesis_kernel_code; }
	virtual const char** get_sum_hypothesis_trace_kernel_code() { return &sum_hypothesis_trace_kernel_code; }
	virtual const char** get_sum_hypothesis_combined_trace_kernel_code() {
		if (use_shared_mem) {
			return &sum_hypothesis_coumbined_trace_kernel_code;
		} else {
			return &sum_hypothesis_coumbined_trace_kernel_code_nosm;
		}
	}

	virtual void allocate_device_memory();
	virtual void create_programs();
	virtual void build_kernel_programs();
	virtual void create_kernels();

	// kernel runner for derived class
	virtual void run_sum_hypothesis_combined_trace_kernel(size_t *global_work_size, size_t *local_work_size)  = 0;
	virtual void run_sum_hypothesis_kernel();
	virtual void run_sum_hypothesis_trace_kernel() = 0;
	virtual void run_sum_trace() = 0;


	void calculate_hypothesis();
	void calculate_sum_hypothesis_trace() {
		run_sum_hypothesis_kernel();
		run_sum_hypothesis_trace_kernel();
	}
	virtual void calculate_correlation_subkey(Array3D<RESULT_T>* corr);
	void calculate_sum_trace() {
		run_sum_trace();
	}

	bool use_shared_mem;

private:
	static cl_platform_id get_target_platform();
	static cl_device_id get_target_device(cl_platform_id platform_id);

	static const char* sum_trace_kernel_code;
	static const char* sum_hypothesis_kernel_code;
	static const char* sum_hypothesis_trace_kernel_code;
	static const char* sum_hypothesis_coumbined_trace_kernel_code;
	static const char* sum_hypothesis_coumbined_trace_kernel_code_nosm;

	bool check_compatibility(cl_device_id device_id, bool need_double);


};


class SOCPAOpenCL : public SOCPAOpenCLBase
{
public:
	SOCPAOpenCL(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model, bool use_shared_mem)
	: SOCPAOpenCLBase(byte_length, num_points, window_size, model, true, use_shared_mem) {
		// create buffers
		allocate_device_memory();

		// create program
		create_programs();

		// build program
		build_kernel_programs();

		// create kernel
		create_kernels();

		clFinish(command_queue);
	};

protected:

	// overrided functions
	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);

	virtual void run_sum_hypothesis_combined_trace_kernel(size_t *global_work_size, size_t *local_work_size);
	virtual void run_sum_hypothesis_trace_kernel();
	virtual void run_sum_trace();

private:


};
#endif //SOCPAOPENCL_H