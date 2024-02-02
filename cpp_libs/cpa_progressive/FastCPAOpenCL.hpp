/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPAOpenCL.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  23-01-2024 16:56:58
*    Last Modified: 02-02-2024 15:46:02
*/

#ifndef FASTCPAOPENCL_H
#define FASTCPAOPENCL_H

#include "FastCPA.hpp"

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

class FastCPAOpenCLBase : public FastCPA
{
public:
	FastCPAOpenCLBase(int num_traces, int num_points, AESLeakageModel::ModelBase *model);
	~FastCPAOpenCLBase();
protected:
	cl_mem cl_device_traces, cl_device_hypothetial_leakage, cl_device_sum_hypothesis, cl_device_sum_hypothesis_square, cl_device_sum_hypothesis_trace;
	cl_context context;
	cl_command_queue command_queue;
	cl_program sum_hypothesis_kernel_program;
	cl_program sum_hypothesis_trace_kernel_program;
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_kernel sum_hypothesis_kernel, sum_hypothesis_trace_kernel;


	virtual const char** get_sum_hypothesis_kernel_code() { return &sum_hypothesis_kernel_code; }
	virtual const char** get_sum_hypothesis_trace_kernel_code() { return &sum_hypothesis_trace_kernel_code; }
	virtual void calculate_hypothesis();
	virtual void run_sum_hypothesis_kernel();
	virtual void run_sum_hypothesis_trace_kernel();

private:
	static cl_platform_id get_target_platform();
	static cl_device_id get_target_device(cl_platform_id platform_id);

	static const char* sum_hypothesis_kernel_code;
	static const char* sum_hypothesis_trace_kernel_code;
};


class FastCPAOpenCL : public FastCPAOpenCLBase
{
public:
	FastCPAOpenCL(int num_traces, int num_points, AESLeakageModel::ModelBase *model);

protected:

	// overrided functions
	virtual void setup_arrays(py::array_t<double> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);
	virtual void calculate_correlation_subkey(Array3D<double>* diff, QUADFLOAT *sumden2);

private:
	bool is_double_available(cl_device_id device_id);

};


#endif //FASTCPAOPENCL_H