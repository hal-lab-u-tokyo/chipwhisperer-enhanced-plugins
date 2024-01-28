/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /cpp_libs/cpa_progressive/FastCPAOpenCLFP32.hpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  28-01-2024 03:41:03
*    Last Modified: 28-01-2024 17:29:01
*/


#ifndef FASTCPAOPENCLFP32_H
#define FASTCPAOPENCLFP32_H

#include "FastCPAOpenCL.hpp"

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


class FastCPAOpenCLFP32 : public FastCPAOpenCLBase
{
public:
	FastCPAOpenCLFP32(int num_traces, int num_points, AESLeakageModel::ModelBase *model);
	~FastCPAOpenCLFP32();

protected:

	virtual const char** get_sum_hypothesis_trace_kernel_code() { return &sum_hypothesis_trace_kernel_code; }

	// overrided functions
	virtual void setup_arrays(py::array_t<double> &py_traces,
						py::array_t<uint8_t> &py_plaintext,
						py::array_t<uint8_t> &py_ciphertext,
						py::array_t<uint8_t> &py_knownkey);
	virtual void calculate_correlation_subkey(Array3D<double>* diff, long double *sumden2);

private:
	Array2D<cl_float2> *traces_df64;
	Array3D<cl_float2> *sum_hypothesis_trace_df64;

	static const char* sum_hypothesis_trace_kernel_code;

	void double_to_float2(const double *src, cl_float2 *dst, int length);
	void float2_to_double(const cl_float2 *src, double *dst, int length);

};

#endif //FASTCPAOPENCLFP32_H