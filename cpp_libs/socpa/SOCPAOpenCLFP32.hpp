/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/SOCPAOpenCLFP32.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  04-05-2025 06:35:09
*    Last Modified: 06-05-2025 08:08:08
*/


#ifndef SOCPAOPENCLFP32_H
#define SOCPAOPENCLFP32_H

#include "SOCPAOpenCL.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


inline cl_float2 to_float2(double x)
{
	const double SPLITTER = (1 << 29) + 1;
	double t = SPLITTER * x;
	double t_hi = t - (t - x);
	double t_lo = x - t_hi;
	return cl_float2({(float)t_hi, (float)t_lo});
}

inline double to_double(cl_float2 x)
{
	return (double)x.x + (double)x.y;
}


class SOCPAOpenCLFP32 : public SOCPAOpenCLBase
{
public:
	SOCPAOpenCLFP32(int byte_length, int num_points, int window_size, AESLeakageModel::ModelBase *model);
	~SOCPAOpenCLFP32();

protected:

	// overrided functions
	virtual void setup_arrays(py::array_t<TRACE_T> &py_traces,
		py::array_t<uint8_t> &py_plaintext,
		py::array_t<uint8_t> &py_ciphertext,
		py::array_t<uint8_t> &py_knownkey);

	virtual void run_sum_hypothesis_combined_trace_kernel(size_t *global_work_size, size_t *local_work_size);
	virtual void run_sum_hypothesis_trace_kernel();
	virtual void run_sum_trace();

	virtual const char** get_sum_trace_kernel_code() { return &sum_trace_kernel_code; }
	virtual const char** get_sum_hypothesis_trace_kernel_code() { return &sum_hypothesis_trace_kernel_code; }
	virtual const char** get_sum_hypothesis_combined_trace_kernel_code() { return &sum_hypothesis_coumbined_trace_kernel_code; }

private:

	// for format conversion
	Array2D<cl_float2> *traces_df64;
	Array3D<cl_float2> *sum_hypothesis_trace_df64;
	Array2D <cl_float2> *sum_trace_x_win_df64, *sum_trace2_x_win_df64;
	Array2D <cl_float2> *sum_trace_x_win2_df64, *sum_trace2_x_win2_df64;
	cl_float2 *sum_hypothesis_combined_trace_df64;

	static const char* sum_trace_kernel_code;
	static const char* sum_hypothesis_trace_kernel_code;
	static const char* sum_hypothesis_coumbined_trace_kernel_code;

	void double_to_float2(const double *src, cl_float2 *dst, int length);
	void float2_to_double(const cl_float2 *src, double *dst, int length);

};



#endif //SOCPAOPENCLFP32_H