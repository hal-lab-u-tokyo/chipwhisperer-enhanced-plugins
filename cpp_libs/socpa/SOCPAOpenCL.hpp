/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPAOpenCL.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  03-05-2025 05:56:44
*    Last Modified: 03-05-2025 08:46:56
*/

#ifndef SOCPAOPENCL_H
#define SOCPAOPENCL_H

#include "SOCPA.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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
	SOCPAOpenCLBase(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model);
	~SOCPAOpenCLBase();
protected:
	// OpenCL device memory
	cl_mem cl_device_traces, cl_device_hypothetial_leakage, cl_device_sum_hypothesis,
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

	size_t local_mem_size;
	size_t global_mem_size;

	virtual const char** get_sum_trace_kernel_code() { return &sum_trace_kernel_code; }
	virtual const char** get_sum_hypothesis_kernel_code() { return &sum_hypothesis_kernel_code; }
	virtual const char** get_sum_hypothesis_trace_kernel_code() { return &sum_hypothesis_trace_kernel_code; }
	virtual const char** get_sum_hypothesis_combined_trace_kernel_code() { return &sum_hypothesis_coumbined_trace_kernel_code; }

	virtual void calculate_hypothesis();
	virtual void calculate_sum_trace();
	virtual void calculate_sum_hypothesis_trace();
	virtual void calculate_correlation_subkey(Array3D<RESULT_T>* corr);

	virtual void run_sum_hypothesis_kernel();

private:
	static cl_platform_id get_target_platform();
	static cl_device_id get_target_device(cl_platform_id platform_id);

	static const char* sum_trace_kernel_code;
	static const char* sum_hypothesis_kernel_code;
	static const char* sum_hypothesis_trace_kernel_code;
	static const char* sum_hypothesis_coumbined_trace_kernel_code;
};


class SOCPAOpenCL : public SOCPAOpenCLBase
{
public:
	SOCPAOpenCL(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model);

protected:

	// overrided functions
	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);
	// 

private:
	bool is_double_available(cl_device_id device_id);

};
#endif //SOCPAOPENCL_H